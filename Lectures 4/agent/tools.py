"""

tools.py that provides tool for the chatbot

each tool has these four items
name  -> unique identifier
description
parameters ->JSON Schema
handler(**kwargs) -> bridge between LLM and Python . It contains the name of the tool

"""
# ─────────────────────────────────────────────────────────
# Safety constants
# ─────────────────────────────────────────────────────────

# Max chars returned from a single read to protect context window.
_READ_CAP_CHARS = 4000

# Max chars accepted by write_file in one call.
_WRITE_CAP_CHARS = 50000

# Cap on grep output to keep context small.
_GREP_MAX_RESULTS = 50
_GREP_MAX_LINE_CHARS = 200

# Default + max timeout for bash commands (seconds).
_BASH_DEFAULT_TIMEOUT = 30
_BASH_MAX_TIMEOUT = 120

#Read_File Tool
def read_file(path:str) ->str:
    #read the file and return the content
    
    #this gives the full path of the file
    try:
        full_path = os.path.abspath(path)
        
        #we open the file in reading mode, encoding utf-8, errors=replace (Which change the undecodable texts to a ? character and dont crash it)
        #we use with because it automatically closes file
        
        with open(full_path,"r",encoding="utf-8",errors="replace") as f:
            data = f.read() #we get data from the file into this variable
            
        #if the data exceeds the context window we truncates it and store it in the variable
        if len(data) > _READ_CAP_CHARS:
             data = data[:_READ_CAP_CHARS] + f"\n... [truncated, total {len(data)} chars]"
        return data
    
    except FileNotFoundError:
        return f"[error] File not found: {path}"
    except PermissionError:
        return f"[error] Permission denied: {path}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"
    
#write_file function
def write_file(path:str,content:str) ->str:
    """
    Creates file with the given content.
    It creates directory if it doesnt exist.
    """
    try:
        if len(content) > _WRITE_CAP_CHARS:
            return f"[error] Content too large: {len(content)} chars > {_WRITE_CAP_CHARS}"
        
        #It does the same work as abspath() and stores the full path
        full = Path(path).resolve()
        
        #Creates Directory
        full.parent.mkdir(parents=True,exist_ok=True)
        full.write_text(content,encoding="utf-8")
        
        return f"[ok] Wrote {len(content)} chars to {full}"

    except PermissionError:
        return f"[error] Permission denied: {path}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"
    
def edit_file(path:str,old_string:str,new_string:str,replace_all: bool=False) ->str:
    """
    Edit file by finding old string and replacing it with new string
    
    it should fail if the old string is not found, old string matching multiple places
    """
    
    try:
        # if old string is empty
        if not old_string:
            return "[error] old_string is empty — refusing to edit (would replace entire file)"
        
        #getting file content and path
        full_path = Path(path).resolve()
        file_content = full_path.read_text(encoding="utf-8",errors="replace") #errors keyword rplaces crashing keywords to ? character so that the program dont crash
        
        #counting occurences
        
        occurences = file_content.count(old_string)
        if occurences == 0:
            return f"[error] old_string not found in {path}"   
        
        #if there are more than one occurance and replace_All is false then throw an error
        if occurences > 1 and not replace_all:
            return (
                f"[error] old_string matches {occurences} times in {path}. "
                f"Make it unique, or pass replace_all=True."
            )
        
        #count = -1 will replace all the occurences
        replaced = file_content.replace(old_string,new_string, -1 if replace_all else 1)
        
        #overwrites the file
        full_path.write_text(replaced,encoding="utf-8")
        
        #now we create a preview
        old_lines = old_string.splitlines() or [""]  #split the lines into a list element or give empty list if the string is empty
        new_lines = new_string.splitlines() or [""]
        diff_preview = []
        
        #we only show the first three  as a preview
        for ln in old_lines[:3]:
            diff_preview.append(f"  - {ln}") #it shows what was removed as -
            
        for ln in new_lines[:3]:
            diff_preview.append(f"  + {ln}") #it shows what waas added as +
            
        if len(old_lines) > 3 or len(new_lines) > 3:
            diff_preview.append("  ...")
            
        scope = f"{occurences} replacement{'s' if (replace_all and occurences > 1) else ''}"
        
        return (
            f"[ok] {scope} in {path}\n"
            + "\n".join(diff_preview)
        )
        
    except FileNotFoundError:
        return f"[error] File not found: {path}"
    except PermissionError:
        return f"[error] Permission denied: {path}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"

def list_dir(path:str =".") ->str:
    """It Lists content of the directory"""
    
    try:
        full = os.path.abspath(path)
        entries = sorted(os.listdir(full)) #listdir() returns name of the files
        if not entries:
            return "Empty Directory."
        
        return "\n".join(entries)
    
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"
    
#glob function that find all the files that matches the glob pattern
def glob(pattern:str,path:str)->str:
    try:
        root = Path(path).resolve()
        matches = sorted(root.glob(pattern))
        
        if not matches:
            return f"(no files match pattern: {pattern})"
        
        #else we show the relative paths for the file 
        relative = [ str(m.relative_to(root)) if m.is_relative_to(root) else str(m) for m in matches]
        
        return "\n".join(relative)
    
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"

#Grep gets where a function aexists in the file and how many times it is ccalled in a file faster than other function
def grep(pattern:str,path:str=".",include:str=""):
    """
    Pattern -> what to find
    path -> where to find ("." means full project)
    include -> inlucde only a particular extension file
    """
    
    #create regex object for the pattern if valid
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"[error] Invalid regex: {e}"
    
    root = Path(path) #this creates a folder type structure
    if not root.exists():
        return f"[error] Path not found: {path}"
    
    #typecasting results list array
    results : list[str] = []
    file_searched = 0
    
    #if we pass a file in grep
    if root.is_file():
        files = [root] #we extract the file part of root
        
    else:
        #if we have included a file type
        if include:
            files = list(root.rglob(include)) #it is recursive glob
            
        else:
            #if we dont include any file type we go through whole project folder
            #skip the junk
            skip_dirs = {".git", ".venv", "__pycache__", "node_modules"}
            
            #we put in file list if the path files doesnt match with skip directory list
            files = [
                p for p in root.glob("*")
                if p.is_file() and not any(part in skip_dirs for part in p.parts)
            ]
            
            #Now we return the results 
            for fpath in files:
                if not fpath.is_file():
                    continue
                    
                file_searched +=1
                
                try: 
                    text = fpath.read_text(encoding="utf-8",errors="replace")
                except Exception:
                    continue
                
                for lineNumber, line in enumerate(text.splitlines(),start=1):
                    #if we find the regex in line
                    if regex.search(line):
                        snippet = line.strip()
                if len(snippet) > _GREP_MAX_LINE_CHARS:
                    snippet = snippet[:_GREP_MAX_LINE_CHARS] + "..."
                
                results.append(f"{fpath}:{lineNumber}:{snippet}")
                
                #if result length exceeded just return it
                if len(results) >= _GREP_MAX_RESULTS:
                    return (
                        "\n".join(results)
                        + f"\n... [truncated, {_GREP_MAX_RESULTS}+ matches; "
                          f"refine pattern to narrow]"
                    )
                    
    if not results:
        return f"(no matches for pattern: {pattern!r} across {file_searched} files)"
    
    return "\n".join(results)

def bash( command:str, timeout: int = _BASH_DEFAULT_TIMEOUT, prompt_fn: Callable[[str],str] |None = None, always_allow: set[str] | None = None,) -> str:
    """
    prompt_fn is for input
    always_allow are the commands tht the user allowed permanently
    
    """
    
    #through cls we will know what is block warn or allow
    cls = safety.classify(command)
    
    if cls.tier == "block":
        #we use the function of safety.py
        return safety.format_block_response(command,cls)
    
    if cls.tier == "warn":
        #We check if it is pre approved or not
        if always_allow not None and command in always_allow:
            pass #do nothing
        
        else:
            #if there is no prompt availble throw an error
            if prompt_fn is None:
                 return (
                    f"[error] WARN command requires user approval but no "
                    f"prompt_fn was provided. Command: {command[:80]}"
                )
                 
            #use safety.py function to format prompt, response adng et final verdict
            prompt = safety.format_prompt(command,cls)
            response = prompt_fn(prompt)
            verdict = safety.parse_permission_response(response)

            if verdict == "deny":
                return "[error] User denied permission for command."
            if verdict == "always":
                #add into allways allow list
                always_allow.add(command)
                
    return execute_bash(command,timeout)
                
    
#Tool Registry

_TOOLS: list[dict] = [
    {
        "name":"read_file",
        "description":(
            "Read the contents of a file. Returns the text (truncated if large). "
            "Use this when you need to see what's in a file."
        ),
        "parameters": {
            "type" : "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (relative or absolute).",
                }
            },
            "required": ["path"],
        },
        
        "handler": read_file,
    },
    {
        "name": "write_file",
        "description": (
            "Create or overwrite a file with the given content. "
            "Creates parent directories if needed. "
            "Use this when creating new files or doing full rewrites."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write.",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content to write.",
                },
            },
            "required": ["path", "content"],
        },
        "handler": write_file,
    },
    {
        "name": "edit_file",
        "description": (
            "Surgical replacement: find an exact `old_string` in the file and "
            "replace it with `new_string`. Fails if old_string is not found or "
            "matches multiple places (unless replace_all=True). "
            "Use this for targeted edits without rewriting the whole file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit.",
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact text to find. Must match exactly once unless replace_all=True.",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement text.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "If true, replace every occurrence. Default false.",
                    "default": False,
                },
            },
            "required": ["path", "old_string", "new_string"],
        },
        "handler": edit_file,
    },
    {
        "name": "list_dir",
        "description": (
            "List files and folders in a directory. "
            "Use this to discover what files exist."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path. Defaults to current dir.",
                    "default": ".",
                }
            },
            "required": [],
        },
        "handler": list_dir,
    },
    {
        "name": "glob",
        "description": (
            "Find files matching a glob pattern. Examples: '*.py', '**/*.md', "
            "'agent/*.py'. Returns paths relative to the search root."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g. '*.py', '**/*.json').",
                },
                "path": {
                    "type": "string",
                    "description": "Root directory to search from. Default '.'.",
                    "default": ".",
                },
            },
            "required": ["pattern"],
        },
        "handler": glob,
    },
    {
        "name": "grep",
        "description": (
            "Search file contents for a regex pattern. Returns "
            "'file:line:matched_line' entries. Use `include` to filter by "
            "file glob (e.g. '*.py'). Skips .git, .venv, __pycache__, node_modules."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search. Default '.' (recursive).",
                    "default": ".",
                },
                "include": {
                    "type": "string",
                    "description": "Optional file glob filter (e.g. '*.py').",
                    "default": "",
                },
            },
            "required": ["pattern"],
        },
        "handler": grep,
    },
    {
        "name": "bash",
        "description": (
            "Run a shell command and return stdout+stderr. "
            "Timeout enforced (default 30s, max 120s). "
            "Use this for: running tests, git operations, installing dependencies, "
            "anything file/shell related that other tools can't do. "
            "Cross-platform: uses bash on Unix/Git-Bash, cmd.exe on Windows. "
            "NOTE: No permission layer yet — be careful with destructive commands."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Default 30, max 120.",
                    "default": 30,
                },
            },
            "required": ["command"],
        },
        "handler": bash,
    },
    {
        "name": "web_search",
        "description": (
            "Search the web using DuckDuckGo. Returns up to N results "
            "(title, URL, snippet). Use this when you need to find "
            "current information, look up documentation, or find resources. "
            "No API key needed. Network call — may be slow or rate-limited."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return (default 8, max 15).",
                    "default": 8,
                },
            },
            "required": ["query"],
        },
        "handler": web_search,
    },
    {
        "name": "web_fetch",
        "description": (
            "Fetch a URL and return its content as clean markdown (LLM-friendly). "
            "Routes through Jina AI which extracts readable text from the page. "
            "Use this AFTER web_search to read the actual content of a result, "
            "or directly when you know which URL you want. "
            "Returns up to ~30K chars. No API key needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL (http:// or https://). https:// is added if missing.",
                },
            },
            "required": ["url"],
        },
        "handler": web_fetch,
    },
    
]


# Public API -> only thing loop import

def get_names() ->list[str]:
    return [t["name"] for t in _TOOLS]

def get_handler(name:str) -> Callable[...,str] | None:
    #it returns a function that return string or NOTHING 
    
    for t in _TOOLS:
        if t["name"] == name:
            return t["handler"]
        
    return None

def get_schema()-> list[dict]:
    #it returns the schema without handler
    
    return [
        {"name": t["name"],"description": t["description"],"parameters":t["parameters"] } for t in _TOOLS
    ]
    