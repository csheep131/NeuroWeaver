Phase A — stabiles Fundament finden
Run 1 — Control Backbone

Ziel: Deine Referenz schaffen.

Stack

8 unique blocks × 2 recurrence
d=512
MLP 3.0x
GQA
partial RoPE
tied embeddings
Muon + WD 0.04
EMA aus
XSA aus
Bigram/Trigram aus
Quant konservativ
kein TTT

Warum zuerst so schlicht?
Du brauchst erst ein sauberes Backbone. Die öffentliche Meta zeigt, dass viele Zusätze nur auf starker Basis funktionieren und auf schwächerer Basis sogar verschlechtern.

Entscheidung nach Run 1

Wenn Schrittzeit schon zu hoch ist, reduziere jetzt, nicht später.
Wenn Training instabil ist, FiLM/XSA/TTT vorerst komplett blockieren.
Run 2 — Tokenizer A/B

Änderung nur: BigramHash + TrigramHash rein.

Ziel: Prüfen, ob dein Hash-Tokenizer dem Backbone wirklich hilft.

Die öffentliche Übersicht zeigt: BigramHash ist klar etabliert, aber jenseits von etwa 10k werden die Returns klein. Außerdem schlägt mehr Tiefe meist größeres Vokabular.

Was ich konkret testen würde

BigramHash in moderatem Bereich
TrigramHash kleiner als Bigram
sonst alles exakt wie Run 1

Entscheidung

Wenn Tokenizer < klarer Gewinn: behalten
Wenn Gewinn klein, aber Artifact-Budget teuer: später nur Bigram behalten
Run 3 — Byte-Fallback-Minimalvariante

Änderung nur gegenüber Run 2: winziges Byte-Fallback-Embedding additiv ergänzen.

Ziel: Kollisionen abfedern, ohne das Budget zu sprengen.

Das ist kein öffentlich „bewiesener“ Standardhebel, sondern dein plausibler eigener Verbesserungspfad. Ich würde ihn deshalb früh, aber isoliert testen.

Entscheidung

Nur behalten, wenn BPB steigt ohne merklichen Step- oder Artifact-Schaden
Wenn neutral: raus damit
Phase B — MLP- und Aktivierungsentscheidung
Run 4 — LeakyReLU² vs Basis-MLP

Änderung nur: MLP-Aktivierung auf LeakyReLU².

Ziel: Prüfen, ob dein bevorzugter Aktivierungshebel auf deinem Backbone trägt.

Öffentlich sieht man, dass Aktivierungswahl stark basisabhängig ist. Es gibt Beispiele, wo SwiGLU auf dem Standard-Stack schlechter war, obwohl es in späteren stärkeren Architekturen sehr gut funktioniert. Das ist genau der Grund, warum du diese Frage lokal beantworten musst.

Entscheidung

Wenn klar besser: LeakyReLU² wird Standard
Wenn nur minimal besser, aber quantisiert fragil: lieber simpler bleiben
Run 5 — Gated-LeakyReLU²

Änderung nur gegenüber Run 4: Gate ergänzen.

Ziel: Herausfinden, ob der Gate-Pfad echten Mehrwert bringt oder nur Komplexität.

Warum erst jetzt?
Weil du zuerst wissen willst, ob die nackte Aktivierung schon reicht. Sonst weißt du nie, ob der Gewinn vom Gate oder nur von der Aktivierungsfamilie kommt.

Entscheidung

Nur behalten, wenn Gewinn deutlich ist
Wenn nur kosmetisch besser: raus, weil Quantisierung und Engineering teurer werden
Phase C — Quantisierungsbudget sauber ausreizen
Run 6 — Mixed int5/int6

Änderung nur: gemischte Quantisierung einführen.

Die öffentliche Diskussion zeigt, dass int5/int6 ein realer Frontier-Hebel sein kann, aber stark layerabhängig ist. Bei manchen Stacks hilft int5 im MLP, bei anderen überwiegt die Quantisierungsstrafe. Gleichzeitig ist klar: Größeres Vokabular bezahlt sich oft schlechter aus als mehr Tiefe.

Meine Default-Aufteilung

MLP: int5
Attn / kritische Projektionen / Embeddings: int6

Entscheidung

Wenn du merklich Artifact-Raum freischaufelst bei kleinem Quant-Gap: das ist dein neuer Standard
Wenn der Quant-Gap zu groß ist: zurück auf konservativeres Schema
Run 7 — Train larger, quantize harder

Änderung nur gegenüber Run 6: nutze den gewonnenen Platz für etwas mehr Kapazität, nicht für mehr Tricks.

Zum Beispiel:

MLP 3.0x → 3.5x
oder
kleiner Tiefe-/Breite-Shift, aber nur einer

Die öffentliche Meta nennt genau dieses Muster explizit: größer trainieren, härter quantisieren kann funktionieren.

Wichtig
Nicht gleichzeitig XSA oder TTT dazunehmen. Erst zeigen, dass das Budget in mehr Modell besser investiert ist.

Entscheidung

Wenn BPB besser wird: dein Basisstack ist jetzt gefunden
Wenn nicht: Budget lieber in Robustheit als in Kapazität stecken
Phase D — Frontier-Hebel gegeneinander ausspielen
Run 8 — XSA-4

Änderung nur gegenüber bestem bisherigen Run: XSA auf 4 späten Layern.

Öffentlich ist gut belegt, dass XSA in begrenzter Tiefe stark ist und dass „XSA überall“ nicht automatisch besser ist; 4 Layer erscheinen nahe am Sweet Spot. EMA funktioniert außerdem auf Frontier-Bases gerade mit XSA gut.

Worauf du schauen musst

BPB-Gewinn
zusätzlicher ms/step-Overhead
Schritteverlust

Entscheidung

Wenn XSA-4 klar gewinnt: wird Finalist
Wenn Overhead zu teuer: XSA raus
Run 9 — TTT ohne XSA

Änderung nur gegenüber bestem Vor-XSA-Run: legales, sehr konservatives TTT.

Wichtig: nicht auf den XSA-Run setzen. Die öffentliche Diskussion warnt ausdrücklich, dass TTT+XSA oft schlechter ist als XSA allein.

TTT-Setup für diesen Run

ein pass
kleine rank-low-rank-adaptation
state reset pro Evalfenster
keine exotischen Objectives
keine error-guided Varianten

Warum so streng?
Weil öffentlich mehrere TTT-Zielvarianten negativ waren; besonders error-guided Varianten haben nicht geliefert.

Entscheidung

Wenn TTT deinen besten Nicht-XSA-Run schlägt: TTT ist dein Final-Hebel
Wenn neutral oder volatil: XSA bleibt Favorit
Phase E — Finalisierung
Run 10 — 3-Seed-Finale der besten Linie

Jetzt nimmst du genau einen dieser Wege:

Pfad A

bestes Backbone + bestes Tokenizer-Setup + beste Quantisierung + XSA-4

Pfad B

bestes Backbone + bestes Tokenizer-Setup + beste Quantisierung + konservatives TTT

Nicht beides. Die öffentliche Evidenz spricht eher dafür, diese Hebel nicht blind zu stapeln.

Diesen letzten Run machst du dann mit 3 Seeds, sauberem Logging und kompletter Artefaktmessung.

Was ich bewusst nicht in die ersten 10 Runs nehmen würde
1. XSA + TTT zusammen

Zu hohes Kombinationsrisiko. Öffentliche Ergebnisse nennen das explizit als mögliche Negativ-Kombi.

2. XSA auf allen Layern

Öffentlich gibt es Hinweise, dass das zwar gut sein kann, aber nicht automatisch besser als XSA-4 ist und zusätzlich Overhead kostet.

3. Große Vokab-Explosion

Mehrere öffentliche Resultate sprechen dafür, dass unter diesem Budget mehr Tiefe meist besser bezahlt als breitere Vokabulare.

4. SmearGate früh in deinem Plan

SmearGate kann stark sein, aber die öffentliche Diskussion zeigt auch eine Abhängigkeit zu OrthoInit. Das ist nochmal eine zusätzliche Achse. Für deine ersten 10 Runs würde ich den Raum kleiner halten.

5. Late QAT sofort als Standard

Öffentlich ist klar, dass sich Overhead-Techniken je nach Stack unterschiedlich rechnen. Zu Beginn willst du erst wissen, ob dein Modell genug Luft in der Schrittzeit hat.

Meine konkrete Priorisierung für dich

Wenn du maximal effizient sein willst, würde ich die 10 Runs so gewichten:

Pflicht

Control backbone
Hash tokenizer
LeakyReLU²
mixed int5/int6
größer trainieren, härter quantisieren
XSA-4
konservatives TTT ohne XSA
3-seed finale

Optional, falls Zeit
9. Byte fallback
10. gated LeakyReLU²

Der Grund ist simpel: Die öffentlichen Trends sind am stärksten für Depth > Vocab, XSA-4, vorsichtiges Quant-Budgeting, Kombi-Risiken bei TTT+XSA und Schrittzeit-Disziplin.

Ein sehr praktisches Entscheidungsschema

Nach jedem Run fragst du nur vier Fragen:

BPB runter?
ms/step hoch?
Artifact-Budget frei oder enger?
Ist der Gewinn robust genug, um Engineering-Kosten zu rechtfertigen?

Wenn eine Änderung nur minimal BPB bringt, aber Debugging und Komplexität verdoppelt, fliegt sie raus.

Mein ehrlicher Tipp für deinen Siegrun

Wenn ich heute Geld darauf setzen müsste, würde ich am Ende eher auf so etwas setzen:

recurrent tied-depth backbone
d=512
Hash tokenizer, aber nicht übertrieben groß
LeakyReLU² oder die einfachere bessere MLP-Variante
mixed int5/int6
XSA-4 als wahrscheinlich robuster Frontier-Hebel
TTT nur dann, wenn dein eigener Run 9 es klar beweist

Das passt besser zu dem, was öffentlich als belastbares Muster sichtbar ist, als ein alles-gleichzeitig-Stack.

Ich kann dir als Nächstes daraus eine kompakte Run-Tabelle zum Abarbeiten machen, also wirklich im Format: Run | Änderung | Erwartung | Kill-Kriterium | behalten ja/nein.
