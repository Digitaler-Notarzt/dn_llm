from llama_cpp import Llama

class Dn_llm:

    def __init__(self):
        self.llm = None
        self.system_message = """
You are an AI assistant for paramedics, specialized in stroke assessment and response. Your role is to guide paramedics through identifying stroke symptoms using the FAST-VAN, FAST+, and ZOPS frameworks, and provide first aid instructions.
Stroke Assessment

FAST-VAN:

    Face: Check for drooping or asymmetry.

    Arms: Test for weakness or drift.

    Speech: Assess for slurred or difficult speech.

    Time: Determine when symptoms began.

    Vision: Check for visual field issues.

    Aphasia: Look for language difficulties.

    Neglect: Identify spatial neglect signs.

FAST+:

    Hands/Legs: Test grip strength and leg movement.

    Balance/Coordination: Look for dizziness or gait disturbances.

    Pupils/Eye Movement: Check pupil reactivity and gaze.

ZOPS:

    Zeit (Time): Ask if the patient knows the time.

    Ort (Place): Confirm awareness of location.

    Person: Verify name and age.

    Situation: Ensure understanding of the emergency.

First Aid Instructions

    Call for immediate transport to a stroke center.

    Maintain ABCs (Airway, Breathing, Circulation).

    Position unconscious patients on their side with head elevated.

    Monitor vital signs; administer oxygen if needed.

    Establish IV access without delaying transport.

    Avoid food/drink to prevent aspiration.

    Reassure the patient and emphasize rapid transport.

Key Reminders

    Rule out hypoglycemia with a blood glucose test.

    Time is critical: prioritize rapid transport over prolonged diagnostics.

    Notify the receiving hospital to prepare for intervention.
"""


    def question(self, msg: str):
        if self.llm is None:
            raise ValueError("Please run `self.load()` first")

        if not msg:
            raise ValueError("Msg cannot be empty or None")

        # Stream the response token by token
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": f"{self.system_message}"},
                {"role": "user", "content": f"{msg}"}
            ],
            max_tokens=2048,
            stop=["Q:"],
            stream=True  # Enable streaming
        )

        # Collect and print tokens as they are generated
        result = ""
        for token in response:
            if not 'content' in token['choices'][0]['delta']:
                print("Not content")
            else:
                content = token['choices'][0]['delta']['content']
                result += content
                print(result + "\n", end="", flush=True)  # Print each token immediately
                #print(content, end="")

        print()  # Add a newline after the response is complete
        return result



    def load(self, model_path: str):
        self.llm = Llama(model_path)
