from agent import Agent

if(__name__ == "__main__"):
    a = Agent(dest_path='test.pth', plotting=False, logs=True)
    a.train()
