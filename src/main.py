from src.environment.OneDimensionalEnvironment import OneDimensionalEnvironment


def main():
    env = OneDimensionalEnvironment()

    full_simulator, obs = env.create_full_simulator()

    print(obs)


if __name__ == "__main__":
    main()
