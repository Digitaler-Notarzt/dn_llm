from src.dn_llm_czlabinger.Dn_llm import Dn_llm

def main():
    llm = Dn_llm()
    llm.load()

    user = input("Frage: ")
    
    while(user != "exit"):
        output = llm.question(msg=f"{user}")
        print(output)
        user = input("Frage: ")

if __name__ == "__main__":
    main()
