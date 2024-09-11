from llama_cpp import Llama

llm = Llama(
   model_path="./capybarahermes-2.5-mistral-7b.Q3_K_S.gguf"
)

def question(msg):
    
    if(msg == "exit"):
        quit()

    output = llm(
        "Q: " + msg + " A: ",
        max_tokens=100,
        stop=["stop", "\n"],
        echo=True
    )
    print(output)

if __name__ == "__main__":
    while(True):
        question(input("Ask a question: "))
