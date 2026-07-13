"""
step 4
loop.py act as REPL (Read Evaluate Print Loop)
"""

from __future__ import annotations
import re
import sys
import json
from typing import Callable #For Type Testing

from agent import llm 
from agent import memory as mem #just created an alias for memory
from agent import tools as tool_registry
from agent import verifier
from agent import rules
from agent.prompts import build_system_prompt

#Maximum Iteration Variable
MAX_ITERATIONS = 8

#Tool Name
_BASH_TOOL = "bash"


#creating function that prints output with flush
def _emit(output_fn: Callable[[str], None], text: str) -> None:
    output_fn(text)
    sys.stdout.flush()
    
def _run_once(
    history: list[dict],
    output_fn: Callable[[str], None],
    input_fn: Callable[[str], str],
    always_allow: set[str],
    plan_mode: bool = False,
) -> bool | tuple:
    """
        Returns:
        True if we got a final answer (turn is done).
        False if we did a tool call (need another iteration).
        ("plan", steps) if the LLM returned a structured plan.

    plan mode true will return plan instead of executing tools
    """
    
    #LLM thinks
    raw = llm.chat(history,system=build_system_prompt(),plan_mode=plan_mode)
    
    #Parse into python dict
    try:
        action = json.loads(raw)
    except json.JSONDecodeError as e:
        _emit(output_fn, f"[error] LLM returned non-JSON: {e}")
        return True  # bail out of this turn
    
    act_type = action.get("action")
    
    #Tool Call
    if act_type == "tool_call":
        tool_name = action.get("tool", "")
        tool_args = action.get("args", {}) or {}
        
        #Through handler we will decide what tool to call
        handler = tool_registry.get_handler(tool_name)
        
        #if there is no hander
        if handler is None:
            observation = f"[error] Unknown tool: {tool_name}"
        else:
            try:
                if tool_name == _BASH_TOOL:
                    observation = handler(
                        prompt_fn=input_fn,
                        always_allow=always_allow,
                        **tool_args,
                    )
                    
                else:
                    observation = handler(**tool_args)
            except Exception as e:
                observation = f"[error] {type(e).__name__}: {e}"
                
        #print for the user after tool 
        _emit(output_fn, f"  🔧 {tool_name}({tool_args})")
        preview = observation if len(observation) < 200 else observation[:200] + "..."
        _emit(output_fn, f"  ✓ {preview}")
        
        #Append tool in history
        history.append(
            {
                "role":"user",
                 "content": f"[Tool Result for {tool_name}]\n{observation}",
            }
        )
        
        return False
    
    #Final Call
    if act_type == "final":
        answer = action.get("answer","")
        _emit(output_fn, f"Agent: {answer}")
        
        mem.add_assistant(history,answer)
        return True
    
    #Planning action
    if act_type == "plan":
        steps =action.get("steps",[])
        #if steps is not a list
        if not isinstance(steps,list) or not steps:
            _emit(output_fn, "[error] Plan action missing 'steps' list.")
            return True
        
        return("plan",steps)

    _emit(output_fn, f"[error] Unknown action type: {act_type}")
    return True

#Planning mode functions
#Printing steps plan
def _print_plan(steps: list[str], output_fn: Callable[[str], None]) -> None:
    _emit(output_fn, "\n  Agent's plan:")
        
    for i, step in enumerate(steps, 1):
        _emit(output_fn, f"    {i}. {step}")
    _emit(output_fn, "")
    
#Asking for allow reject or edit
def _handle_plan_mode(
    history: list[dict],
    output_fn: Callable[[str], None],
    input_fn: Callable[[str], str],
    always_allow: set[str],
    goal: str = "",
    verifier_mode: bool = False,
) -> None:
    """
    y -> call execute plan mode
    n -> reject plan
    edit -> get feedback append 
    
    """
    while True:
        #get plan IF EXIST
        result = _run_once(history, output_fn, input_fn, always_allow, plan_mode=True)
        
        #result[0] have type , result[1] have steps if it is a plan        
        
        #If run once dont return plan tuple
        if not (isinstance(result,tuple) and result[0] == "plan"):
            return
        
        steps = result[1]
        _print_plan(steps,output_fn)
        
        #Now input for y n e
        try:
            response = input_fn("  [Plan mode] Proceed? [y/n/edit]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            _emit(output_fn, "\n  Plan cancelled.")
            return

        if response in ("y", "yes"):
            _execute_plan_steps(
                history, steps, output_fn, input_fn, always_allow,
                goal=goal, verifier_mode=verifier_mode,
            )
            return
        
        elif response in ("edit", "e"):
            try:
                feedback = input_fn("  Edit feedback: ").strip()
            except (EOFError, KeyboardInterrupt):
                _emit(output_fn, "\n  Plan cancelled.")
                return
            history.append({
                "role": "user",
                "content": f"User feedback on your plan: {feedback}",
            })
            _emit(output_fn, "  Re-planning with your feedback...\n")
            continue  # loop back to get revised plan
        else:
            _emit(output_fn, "  Plan rejected.\n")
            return
        
        

def _execute_plan_steps(
    history: list[dict],
    steps: list[str],
    output_fn: Callable[[str], None],
    input_fn: Callable[[str], str],
    always_allow: set[str],
    goal: str = "",
    verifier_mode: bool = False,
    max_verify_retries: int = 10,
) -> None:
    
    """
    Main Worker FUnction for executing planning
    
    verifier = True -> call verifier
    
    
    verifier = False -> append to history and re-execute the plan

    """
    
    verify_attempt = 0
    while True:
        total = len(steps)
        for i, step in enumerate(steps, 1):
            _emit(output_fn, f"\n  ── Step {i}/{total}: {step} ──")    
        
            history_len_before = len(history)
            tool_called = False
            step_finished = False
            
            
            history.append({
                "role": "user",
                "content": (
                    f"[Plan step {i}/{total} of approved plan]\n"
                    f"Task: {step}\n\n"
                    f"REQUIRED: Call the appropriate tool NOW (read_file, "
                    f"write_file, edit_file, bash, etc.) — do NOT just describe. "
                    f"After the tool result is back and this step is complete, "
                    f"return {{\"action\": \"final\", \"answer\": \"<one-line summary>\"}}."
                ),
            })
            
            for iteration in range(MAX_ITERATIONS):
                result = _run_once(history, output_fn, input_fn, always_allow, plan_mode=False)
                if isinstance(result, tuple):
                    # Plan-mode unexpected here, bail.
                    break

                # Detect if a tool was called during this iteration.
                if len(history) > history_len_before:
                    tool_called = True

                if result:  # True = final/error → step done
                    step_finished = True
                    break
            
            #LLM should call a tool first if it hasnt
            if step_finished and not tool_called:
                _emit(output_fn, f"  ⚠ Step {i} skipped tool use — retrying with reminder.")
                retry_history_len = len(history)
                history.append({
                    "role": "user",
                    "content": (
                        f"[Retry: Step {i}/{total}]\n"
                        f"You returned a final answer without calling any tool. "
                        f"Task: {step}\n\n"
                        f"You MUST call a tool (read_file/write_file/edit_file/bash) "
                        f"BEFORE answering. Use the tool to actually do the work. "
                        f"Then return final with a one-line summary."
                    ),
                }) 
                
                for _ in range(MAX_ITERATIONS):
                    result = _run_once(history, output_fn, input_fn, always_allow, plan_mode=False)
                    if isinstance(result, tuple):
                        break
                    if len(history) > retry_history_len:
                        tool_called = True
                    if result:
                        break
                if not tool_called:
                    _emit(output_fn, f"  ⚠ Step {i} retry also skipped tools.")
            elif not step_finished:
                _emit(output_fn, f"  ⚠ Step {i} hit MAX_ITERATIONS={MAX_ITERATIONS}.")

        _emit(output_fn, "\n  ✓ Plan steps executed.\n")
        
        #if verifier phase is not enabked
        if not verifier_mode or not goal:
            return
        
        verify_attempt += 1
        if verify_attempt > max_verify_retries:
            _emit(output_fn, f"  ⚠ Verifier retries exhausted ({max_verify_retries}).\n")
            return

        verdict = verifier.verify(goal, history, emit=output_fn)

        if verdict.passed:
            _emit(output_fn, "  ✓ Plan completed & verified.\n")
            return

        # FAIL: feed coach feedback back as a user message and re-execute the plan.
        _emit(output_fn, f"  ⟳ Re-running plan with verifier feedback (attempt {verify_attempt + 1})...\n")
        feedback = (
            f"[Verifier feedback — round {verify_attempt}]\n"
            f"{verdict.to_feedback_message()}\n\n"
            f"Re-execute the same plan with this guidance. "
            f"When done, return final with a one-line summary."
        )
        history.append({"role": "user", "content": feedback})
        # loop continues: re-execute the same plan with feedback in context.
    
    

#Main function that does all the tasks
def run(input_fn: Callable[[str],str] = input,  output_fn:Callable[[str],None] = print):
    
    #We put default input function into input_fn variable works like input function that TAKES STR and RETURNS STR
    #We put default input function into output_fn variable works like print function that TAKES STR and RETURNS NOTHING
    
    #Initilazing History with system prompt
    history =  mem.init_history(build_system_prompt)
    
    always_allow: set[str] = set()
    
    _emit(output_fn, "╔════════════════════════════════════════════╗")
    _emit(output_fn, "║   Demo Harness — Checkpoint 2              ║")
    _emit(output_fn, "║   Mock LLM + Tools (ReAct loop)            ║")
    _emit(output_fn, "║   Tools: " + ", ".join(tool_registry.list_names()) + "  ║")
    _emit(output_fn, "║   Type 'quit' or 'exit' to leave.          ║")
    _emit(output_fn, "╚════════════════════════════════════════════╝")
    _emit(output_fn, "")
    
    #Inifnite looping
    
    while True:
        #user input
        try:
            user_text = input_fn("You: ")
        except (EOFError, KeyboardInterrupt):
            _emit(output_fn, "\nBye!")
            return
        
        user_text = user_text.strip()
        #If User Doesnt Write Anything
        if not user_text: 
            continue
        
        #If the user write quit
        if user_text.lower() in ("quit", "exit", "q"):
            _emit(output_fn, "Bye!")
            return
        
        # /plan toggle (session-scoped)
        if user_text.lower() == "/plan":
            plan_mode = not plan_mode
            state = "ON" if plan_mode else "OFF"
            _emit(output_fn, f"\n  ☐ Plan mode: {state}\n")
            continue
        
        # /verifier toggle (session-scoped)
        if user_text.lower() == "/verifier":
            verifier_mode = not verifier_mode
            state = "ON" if verifier_mode else "OFF"
            _emit(output_fn, f"\n  ☐ Verifier mode: {state}\n")
            continue
        
        # /log_mistake 
        if re.match(r"^/log_mistake(?:\s|$)", user_text, re.IGNORECASE):
            from agent import mistakes
            description = user_text[len("/log_mistake"):].strip()
            

            if description:
                # Manual mode: user typed free-form text.
                rule = rules.log_mistake(description) 
                _emit(output_fn, f"  ✓ Logged as [{rule.id}] in DemoHarness.md\n")
                continue
            
            # Auto-detect mode: scan recent history for mistake signals.
            detected = mistakes.detect_mistakes(history)
            if not detected:
                _emit(output_fn, "  No mistakes detected in recent history.\n")
                continue

            _emit(output_fn, f"  Detected {len(detected)} mistake signal(s):\n")
            for m in detected:
                _emit(output_fn, f"    • [{m.source}] {m.why}\n")
                
            #extraction of rules
            candidates = mistakes.extract_rules(detected)
            if not candidates:
                _emit(output_fn, "  No generalizable rules emerged.\n")
                continue

            _emit(output_fn, f"  Extracted {len(candidates)} rule(s):\n")
            for c in candidates:
                rule = rules.log_rule(
                    text=c["text"],
                    why=c.get("why", ""),
                    category=c.get("category", "general"),
                )
                _emit(output_fn, f"  ✓ [{rule.id}] {c['text']}\n")
            continue
        
        #recall keyword ->searcg and show matching rules
        if re.match(r"^/recall(?:\s|$)", user_text, re.IGNORECASE):
            query = user_text[len("/recall"):].strip()
            
            if not query:
                _emit(output_fn, "  Usage: /recall <keyword>\n")
                continue
            matches = rules.recall(query)
            if not matches:
                _emit(output_fn, f"  No rules matching '{query}'.\n")
                continue
            
            _emit(output_fn, f"  Found {len(matches)} rule(s) for '{query}':\n")
            for r in matches:
                _emit(output_fn, f"\n  ── [{r.id}] {r.title} ──\n")
                
                #first non-metadata line is trimmed
                for line in r.body.splitlines():
                    stripped = line.strip()
                    if stripped and not stripped.startswith("**Logged:"):
                        _emit(output_fn, f"  {stripped}\n")
                        break
            continue
        
        #If list the nshow all the rules

        # /list — show all rules (compact view).
        if user_text.lower() == "/list":
            all_rules = rules.list_rules()
            if not all_rules:
                _emit(output_fn, "  No rules yet. Use /log_mistake to add one.\n")
                continue
            _emit(output_fn, f"  {len(all_rules)} rule(s) in DemoHarness.md:\n\n")
            for r in all_rules:
                _emit(output_fn, f"  • {r.short()}\n")
            continue

        #Updating Memory
        mem.add_user(history,user_text)
        goal = user_text
        
        #Plan Mode
        if plan_mode:
            _handle_plan_mode(
                history, output_fn, input_fn, always_allow,
                goal=goal, verifier_mode=verifier_mode,
            )
            continue
        
        
        #tools until maximum interation or finished
        # Iterate LLM <-> tools until final answer or cap.
        for i in range(MAX_ITERATIONS):
            done = _run_once(history, output_fn, input_fn, always_allow)
            if done:
                break
        else:
            _emit(output_fn, f"[warn] Hit MAX_ITERATIONS={MAX_ITERATIONS}, stopping.")
        # Verifier phase (single task, no plan).
        if verifier_mode:
            for attempt in range(1, 11):  # up to 10 retries
                verdict = verifier.verify(goal, history, emit=output_fn)
                if verdict.passed:
                    _emit(output_fn, "  ✓ Verified.\n")
                    break
                _emit(output_fn, f"  ⟳ Verifier said: {verdict.reason}\n")
                feedback = (
                    f"[Verifier feedback — attempt {attempt}]\n"
                    f"{verdict.to_feedback_message()}"
                )
                history.append({"role": "user", "content": feedback})
                for j in range(MAX_ITERATIONS):
                    done = _run_once(history, output_fn, input_fn, always_allow)
                    if done:
                        break
                else:
                    _emit(output_fn, f"[warn] Hit MAX_ITERATIONS={MAX_ITERATIONS} during verify retry.")
                    break
            else:
                _emit(output_fn, "  ⚠ Verifier retries exhausted.\n")
                
                    
        #LLM thinks
        try:
            reply=llm.chat(history)
        except Exception as e:
            output_fn(f"[Agent error] {e}")
            continue
        
        #Save assitant reply
        mem.add_assistant(history,reply)
        
        #Printing
        output_fn(f"Agent: {reply}")
        sys.stdout.flush()