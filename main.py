from src.dn_llm_czlabinger.Dn_llm import Dn_llm

def main():
    llm = Dn_llm()
    output = llm.question(msg="How are you feeling?", system_message="You are a AI model you dont have feelings.")

    print(output)

if __name__ == "__main__":
    main()
