Usage:
```py
from dn_llm_czlabinger.Dn_llm import Dn_llm

llm = Dn_llm()
llm.load()
output = llm.question(msg="Mein Patient hat einen Schalganfall! Was soll ich machen?")

print(output)
```


TODO: 
- [X] Systemprompt to ignore off topic stuff
- [X] Output Token limit
