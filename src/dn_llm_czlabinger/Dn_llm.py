from llama_cpp import Llama

class Dn_llm:

    def __init__(self):
        self.llm = Llama.from_pretrained(
            repo_id="akjindal53244/Llama-3.1-Storm-8B-GGUF",
            filename="Llama-3.1-Storm-8B.Q4_K_M.gguf",
        )

    def question(self, msg: str, system_message: str):
            if(msg == "exit"):
                quit()


            output = self.llm.create_chat_completion(
                  messages = [
                      {"role": "system", "content": f"{system_message}"},
                      {
                          "role": "user",
                          "content": f"{msg}"
                      }
                  ]
            )

            print(output)
