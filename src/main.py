from src.one_dimensional.WaveWorld import WaveWorld


def main():

    wave_world = WaveWorld(world_size=7, local_simulation=False)

    for _ in range(10):
        wave_world.step(0, verbose=True)


if __name__ == "__main__":
    main()
