from src.dn_llm.Dn_llm import Dn_llm

def main():
    llm = Dn_llm()
    llm.load("/home/stoffi05/Downloads/Repo-8.0B-F16.gguf")

    user = input("Frage: ")
    
    while(user != "exit"):
        output = llm.question(msg=f"{user}")
        print(output)
        user = input("Frage: ")

if __name__ == "__main__":
    main()
