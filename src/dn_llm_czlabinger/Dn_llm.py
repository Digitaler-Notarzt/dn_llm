from llama_cpp import Llama


def setup(self):
    self.llm = Llama.from_pretrained(
        repo_id="akjindal53244/Llama-3.1-Storm-8B-GGUF",
        filename="Llama-3.1-Storm-8B.Q4_K_M.gguf",
    )

def question(self):
    while True:
        msg = input("Ask a question: ")
            
        if(msg == "exit"):
            quit()

        output = self.llm(
            "Q: " + msg + " A: ",
            max_tokens=100,
            stop=["stop", "\n"],
            echo=True
        )
        print(output)
    
