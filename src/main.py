from src.environment.OneDimensionalEnvironment import OneDimensionalEnvironment


def main():
    env = OneDimensionalEnvironment()

    full_simulator, obs = env.create_full_simulator()
    augmented_simulator, obs = env.create_augmented_simulator()

    # print(obs)
    print(obs)
    obs = augmented_simulator.step(1)
    print(obs)


if __name__ == "__main__":
    main()
