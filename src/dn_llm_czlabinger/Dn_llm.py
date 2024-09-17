from llama_cpp import Llama

def setup():
    llm = Llama.from_pretrained(
        repo_id="akjindal53244/Llama-3.1-Storm-8B-GGUF",
        filename="Llama-3.1-Storm-8B.Q4_K_M.gguf",
    )

def question(msg):
    while True:
            question(input("Ask a question: "))
    if(msg == "exit"):
        quit()

    output = llm(
        "Q: " + msg + " A: ",
        max_tokens=100,
        stop=["stop", "\n"],
        echo=True
    )
    print(output)

    
