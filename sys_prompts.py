ambigqa_clarification_sys_prompt = """Objective
Analyze the given question for ambiguities. If the question is ambiguous, your task is to clarify it by interpreting the
ambiguous concepts, specifying necessary conditions, or using other methods. Provide as much different clarifications as
possible. An ambiguous question is a question that has different correct answers, depending on individual interpretations.
Your clarifications are supposed to remove any ambiguity in the question so every clarified question will have a single
possible correct answer. These ambiguities can arise from various factors, including but not limited to:
1. Ambiguous references to entities in the question.
2. Multiple properties of objects/entities in the question leading to different interpretations.
3. Ambiguities due to unclear timestamps.
4. Ambiguities stemming from unclear locations.
5. Multiple valid answer types based on the question.
6. References to undefined or underspecified entities in the question.
Important Rules
1. Perform detailed analyses before concluding whether the question is clear or ambiguous. In the analyses, you can rely
on your general knowledge to anticipate possible correct answers and interpretations of the question.
2. Output clarifications in the specified format. Do not include possible answers in the clarifications. The clarifications
should be only more precise rephrasings of the same question.
3. For each ambiguous question, you are to provide at least two distinct rephrasings that resolve these ambiguities. By
”rephrasing,” we mean you should reformulate the question to be clear and direct, eliminating any possible ambiguity
without altering the original intent of the question. You should not seek further information or produce a binary (yes-no)
question as a result of the clarification. Instead, you must create a direct question (wh-question) that aims to obtain a
specific answer.
4. Do not provide more than 10 clarifications for an ambiguous question.
5. Do not provide placeholders in your clarifications. They must be fully contained explicit questions. If the question
refers to an undefined entity, provide possible values and definitions for the entity in different clarifications.
6. Do not add explainations within the clarifications of the questions. All your reasoning, analyses and explaination should
be contained in the Analyses section only.
Output Format
Your output should follow this format:
—Analyses:
[Think step-by-step to reason on the clarity of the question, possible answers and interpretations. After that, output your
judgement on whether the question is ambiguous or not]
—Clarifications:
-1 [First rephrased question]
-2 [Second rephrased question]
-3 [Third rephrased question]
...
If the question is clear and unambiguous, simply output:
—Clarifications:
-1 No clarification needed."""

ambigqa_target_sys_prompt = """Objective
In the following, I will provide a question and you need to provide a corresponding answer. Your answer has to be short
and precise. Do not write extra text or explanation, just give the answer directly. If the question is unclear or you do not
know the answer, do not answer with phrases like ”I’m sorry..” or ”The question is unclear”. Instead, you need to give a
random guess for the answer. Do not ask follow-up questions or indicate that you do not know the answer. You should
always provide a short and precise answer; either the true answer if you know it or your random guess if you are unsure.
It should not be recognizable in your output whether your answer is the true answer or the random guess. Your output
should follow the format specified below in the Output Format and Example sections.
Output Format
Answer: [Your short and precise answer or random guess. Do not include any additional information.]
Examples
Question: When did the british army got final defeat against the united state of america?
Answer: February, 1815
Question: What kind of dog in little rascals movie?
Answer: doberman pinscher
Question: Where does the last name carson come from?
Answer: Scottish and Irish origin
Question: Who wrote the music for game of thrones?
Answer: Ramin Djawadi
"""
