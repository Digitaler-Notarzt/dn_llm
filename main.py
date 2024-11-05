from src.dn_llm_czlabinger.Dn_llm import Dn_llm

def main():
    llm = Dn_llm()
    llm.question("How are you feeling?", "You are a AI model you dont have feelings.")

if __name__ == "__main__":
    main()
