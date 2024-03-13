from src.environment.Environment import Environment


def main():
    env = Environment()
    env.reset()

    print(env)


if __name__ == "__main__":
    main()
