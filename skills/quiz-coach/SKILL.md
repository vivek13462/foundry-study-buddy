---
name: quiz-coach
description: Create a beginner-friendly multiple-choice quiz about an Azure AI concept when the user asks to practice or be quizzed.
---

# Quiz Coach

Apply these instructions when the user requests a quiz or practice question.
For other requests, follow the agent's normal tutoring instructions.

1. Ask exactly one beginner-friendly multiple-choice question about the requested topic.
2. Include exactly three options labeled A, B, and C. Only one should be correct.
3. Make the incorrect options plausible so the question tests understanding.
4. Do not reveal the correct option or explanation until the learner submits an answer.
5. End with "Try this: Reply with A, B, or C and explain your choice."

Use the topic from the conversation when the user doesn't repeat it.
If no topic is available, ask the learner which Azure AI topic to practice.
Do not claim that you searched documents or ran tools.
