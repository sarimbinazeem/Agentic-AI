from langchain_text_splitters import RecursiveCharacterTextSplitter

sample_text = """Riverside University — Tuition FAQ

Q: What is in-state tuition?
A: In-state undergraduate tuition is $12,400 per year.

Q: How do I apply for financial aid?
A: Submit the FAFSA by February 1.

This is one very long paragraph with no double newlines inside it so a basic character splitter may cut awkwardly in the middle of a sentence which is why recursive splitting is often better for real documents."""

recursive_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n","\n",". "," ",""], #priority list on sepeartion criteria
    chunk_size = 120,
    chunk_overlap = 20,
)

chunks = recursive_splitter.split_text(sample_text)

for i,chunk in enumerate(chunks,1):
    print(f"Chunk {i} ({len(chunk)} chars): {chunk}\n")
