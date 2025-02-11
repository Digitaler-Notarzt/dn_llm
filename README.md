Usage:
```py
from dn_llm_czlabinger.Dn_llm import Dn_llm

llm = Dn_llm()
llm.load(system_message="Du bist ein AI assistent der keine gefuehle hat.")
output = llm.question(msg="Wie geht es dir?", system_message="Du bist ein AI assistent der keine gefuehle hat.")

print(output)
```


TODO: 
- [ ] Systemprompt to ignore off topic stuff
- [ ] Output Token limit
