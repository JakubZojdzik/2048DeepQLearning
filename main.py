from agent import Agent

if(__name__ == "__main__"):
    a = Agent(source_path='model.pth', dest_path='model.pth')
    a.train(num_episodes=1000)
