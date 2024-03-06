from src.environment.OneDimensionalEnvironment import OneDimensionalEnvironment
import numpy as np

def main():
    env = OneDimensionalEnvironment(seed=43)

    full_simulator, obs = env.create_full_simulator()
    augmented_simulator, obs = env.create_augmented_simulator()

    for _ in range(10):
        print(augmented_simulator.step(np.random.choice([-1, 1])))


if __name__ == "__main__":
    main()
