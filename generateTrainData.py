from openai import OpenAI

def request():
    client = OpenAI(
        api_key="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"
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": """Bitte schreibe mir einen Dialog zwischen einem Rettungssanitäter und einer AI welche ihm helfen soll einen Schlaganfall bei einem Patienten zu diagnostizieren.
Verwende dazu folgende Hilfen zusaetzlich zum FAST-Test:
Wie erkennt man einen Schlaganfall:

Wichtig als Sani: man muss Blutzucker messen!!! Bei einer Unterzuckerung können Patienten ebenso Schlaganfallsymptome aufweisen, da die Zuckerreserven in den Zellen bereits aufgebraucht wurde ==> Bei Diabetikern sollte man dies besonders beachten, da es dazu kommen kann, dass zB zu viel Insulin gespritzt wurde.

Eigenanamnese:

Du hast verschiedene Schemen, wie FAST+ und ZOPS

Face: hängt ein Mundwinkel, kann ein Augenlid evtl nicht geschlossen werden? Abnormalitäten im Gesicht
Arms: Kann der Patient die Arme nicht heben? Wie testet man das: Patient soll die Arme in die Höhe geben, die Handflächen nach oben schauen lassen. Der Patient soll anschließend die Augen schließen. Somit kann man ebenso sehr gut Anzeichen eines Insults erkennen.
Speech: Verschwommene Sprache? Kann der Patient plötzlich Sachen nicht mehr aussprechen?
Zeitlicher Verlauf: Hatte der Pat evtl schon einmal einen Insult und kann dies deswegen nicht bewegen, oder sind diese Sachen neu aufgetreten? Hierbei ist ebenso wichtig zu erfahren, wie lang dies bereits ist und vor allem ist es wichtig, dass der Pat so schnell wie möglich behandelt wird ==> Gehirnzellen sterben nach ca 3 Minuten ohne Versorgung ab, da die Fettreserven in den Zellen sehr gering sind

Das Plus beim Fast: Hände, Beine

Hände: Kann der Patient die Hände zudrücken? Der Pat soll dem Sanitäter die Hände geben und mit beiden Händen so fest wie möglich zudrücken. Gibt es evtl unterschiede in der Druckkraft? Kann der Pat überhaupt zudrücken?

Beine: Weist der Patient eine Beinschwäche auf?

Hat der Patient Schwindel bzw Gleichgewichtsprobleme, die davor noch nicht da waren

Sieht der Patient Doppelbilder

Wie schauen die Pupillen aus? Reagieren die Pupillen auf Licht und sind gleich groß?

Herdblick? Können Personen ihre Umgebung nur mehr eingeschränkt wahrnehmen?
ZOPS:

Zeit: ist der Pat zeitlich orientiert?
Ort: ist der Patient örtlich orientiert?
Person: weiß der Pat, wie er heißt bzw. wie alt er ist?
Situation: Was ist passiert? Warum wurde die Rettung gerufen?

Fremdanamnese:

Es ist immer wichtig zu erfahren, ob sich der Pat im Wesen bzw. iwie verändert hat. Des Weiteren ist es wichtig, dass man als Sani herausfindet, ob ein Insult bereits einmal diagnostiziert wurde und ob der Pat bereits dort bleibende Schäden davongetragen hat

Gib die Daten in folgendem Format auts:
**RS** Nein, der Mundwinkel hängt nicht, und beide Augenlider funktionieren normal. **KI** Gut. Bitte überprüfe die Arme: Kann der Patient beide Arme heben und halten? **RS** Ja, er kann beide Arme problemlos heben und halten. Keine Schwäche oder Abweichung erkennbar. **KI** Wie ist die Sprache des Patienten? Gibt es Probleme mit verwaschener Sprache oder Schwierigkeiten beim Sprechen? **RS** Seine Sprache ist klar und verständlich. Er kann ohne Probleme vollständige Sätze bilden. **KI** Das klingt bisher nicht nach einem Schlaganfall. Wann haben die Symptome begonnen und wie haben sie sich entwickelt? **RS** Der Patient sagt, dass die Symptome heute Morgen beim Aufstehen begonnen haben. Der Schwindel wurde schlimmer, als er den Kopf drehte. **KI** Interessant. Lass uns weitere Tests durchführen. Kann der Patient deine Hände fest zudrücken? Gibt es Unterschiede in der Kraft? **RS** Beide Hände haben eine gleich starke Druckkraft, keine Auffälligkeiten hier. **KI** Und wie sieht es mit den Beinen aus? Kann er beide Beine ohne Schwäche bewegen? **RS** Ja, er kann beide Beine normal bewegen. Allerdings sagt er, dass er sich unsicher fühlt beim Gehen wegen des Schwindels. **KI** Verstanden. Überprüfe bitte die Pupillen: Sind sie gleich groß und reagieren sie normal auf Licht? **RS** Die Pupillen sind gleich groß und reagieren normal auf Licht. Keine Auffälligkeiten hier. **KI** Gut. Hat der Patient Übelkeit oder Erbrechen? **RS** Ja, er klagt über starke Übelkeit und hat einmal erbrochen. **KI** Wird der Schwindel durch Kopfbewegungen verstärkt? **RS** Ja, er sagt, dass der Schwindel deutlich schlimmer wird, wenn er den Kopf dreht oder sich hinlegt.
**Result** Die Symptome – Schwindel, Übelkeit, Kopfschmerzen, Verstärkung bei Kopfbewegungen – deuten nicht auf einen Schlaganfall hin.

Variire ob es sich um einen Schlaganfall handelt oder nicht.

""",
            }
        ],
        model="gpt-4o",
    )

    message = chat_completion.choices[0].message.content.replace('\\n', '').replace('\n', '')

    first_rs_index = message.find('**RS**')
    if first_rs_index != -1:
        second_rs_index = message.find('**RS**', first_rs_index + 1)
        if second_rs_index != -1:
            # Entferne alles vor dem zweiten **RS**
            message = message[second_rs_index:]

    # Teile die Nachricht in zwei Teile auf
    result_index = message.find('**Result**')
    if result_index != -1:
        before_result = message[:result_index]
        after_result = message[result_index:]

        # Finde den Index des ersten ---, **RS** oder **KI** nach **Result**
        end_index = after_result.find('---')
        if end_index == -1:
            end_index = after_result.find('**RS**')
        if end_index == -1:
            end_index = after_result.find('**KI**')

        if end_index != -1:
            after_result = after_result[:end_index]

        # Schreibe die Teile in separate Dateien
        with open('dialogs.txt', 'a', encoding='utf-8') as file:
            print(f"Wrote {before_result} to dialogs.txt\n\n")
            file.write(before_result + '\n')

        with open('responses.txt', 'a', encoding='utf-8') as file:
            after_result = after_result.replace("**Result** ", '')
            print(f"Wrote {after_result} to results.txt")
            file.write(after_result + '\n')
    else:
        print("Kein **Result** gefunden.")

def main():
    for i in range(0, 200):
        print(f"Sending request number {i}")
        request()

if __name__ == "__main__":
    main()
