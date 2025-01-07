from src.dn_llm_czlabinger.Dn_llm import Dn_llm

def main():
    llm = Dn_llm()
    llm.load(system_message="Du bist ein AI assistent der keine gefuehle hat.")
    output = llm.question(msg="Wie geht es dir?", system_message="Du bist ein AI assistent der keine gefuehle hat.")

    print(output)

if __name__ == "__main__":
    main()
