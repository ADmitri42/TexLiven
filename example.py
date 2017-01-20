from network import Texliven
TexLivenMind = Texliven()

while True:
    quest = input(">")
    answer = TexLivenMind.generate_answer(quest)
    print(answer)
