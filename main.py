#!/usr/bin/env python3
# -*- coding: ascii -*-

import argparse
import functools
import json
import logging
import multiprocessing
import numpy as np
import os
import scipy.interpolate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-templates",
        type=int,
        default=4,
        help="number of templates",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=360,
        help="number of optimization iterations",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=12,
        help="size of each generation",
    )
    parser.add_argument(
        "--fertility",
        type=int,
        default=60,
        help="number of offspring made by each generation",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        help=".npy file to write resulting templates to",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="seed to control for stochasticity",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="don't plot the final narrative arc template curves",
    )
    args = vars(parser.parse_args())
    if not args["no_plot"]:
        global plt, sns
        import matplotlib.pyplot as plt
        import seaborn as sns
    assert args["num_templates"] > 0
    assert args["num_generations"] > 0
    assert args["population_size"] > 0
    if args["outfile"] is not None and os.path.exists(args["outfile"]):
        logging.warning("{} exists; overwriting".format(args["outfile"]))
    return args


def scale(
    value: float, start_min: float, start_max: float, end_min: float, end_max: float
) -> float:
    """Returns the result of scaling value from the range
    [start_min, start_max] to [end_min, end_max].
    """
    return end_min + (end_max - end_min) * (value - start_min) / (start_max - start_min)


def load_data(filename):
    with open(filename, "r") as infile:
        raw_data = json.load(infile)
    logging.info("successfully parsed json in {}".format(filename))
    training, validation, test = list(), list(), list()
    stats = {"min_track_number": float("inf"), "max_track_number": 0}
    while raw_data:
        raw_album = raw_data.pop()
        min_album_track, max_album_track = float("inf"), 0
        album = list()
        for raw_track in raw_album:
            album.append((raw_track["track number"], raw_track["valence"]))
            min_album_track = min(min_album_track, raw_track["track number"])
            max_album_track = max(max_album_track, raw_track["album tracks"])
            max_album_track = max(max_album_track, raw_track["track number"])
        for i, (track_number, valence) in enumerate(album):
            album[i] = (
                scale(
                    track_number,
                    min(1, min_album_track),
                    max_album_track,
                    0,
                    1,
                ),
                (valence - np.mean([v[1] for v in album]))
                / np.std([v[1] for v in album]),
            )
        if raw_album[0]["set split"] == "training":
            training.append(album)
        elif raw_album[0]["set split"] == "validation":
            validation.append(album)
        else:
            assert raw_album[0]["set split"] == "test"
            test.append(album)
        stats["min_track_number"] = min(stats["min_track_number"], min_album_track)
        stats["max_track_number"] = max(stats["max_track_number"], max_album_track)
    return (training, validation, test), stats


def init_templates():
    return np.zeros((args["num_templates"], 7))


def expand_templates(tpls):
    return [
        scipy.interpolate.interp1d(
            [0.0, 0.2, 0.3, 0.5, 0.65, 0.8, 1.0],
            tpls[i, :],
            kind="cubic",
        )
        for i in range(tpls.shape[0])
    ]


def fitness(tpls, dataset):
    expanded_tpls = expand_templates(tpls)
    rv = 0
    for album in dataset:
        best = float("inf")
        for tpl in expanded_tpls:
            error = 0
            for track_number, valence in album:
                error += np.square(tpl(track_number) - valence)
            error /= len(album)
            if error < best:
                best = error
        rv += best
    return rv / len(dataset)


def crossover(population):
    children = list()
    for _ in range(args["fertility"]):
        father_idx, mother_idx = np.random.choice(
            len(population), size=2, replace=False
        )
        father, mother = population[father_idx], population[mother_idx]
        child = np.zeros_like(father)
        for i in range(child.shape[0]):
            mask = np.random.randint(2, size=child.shape[1])
            child += father * mask
            child += mother * (1 - mask)
        children.append(child)
    return children


def mutate(tpls):
    return tpls + np.random.normal(0, np.random.rand(), size=tpls.shape)


def optimize(population, dataset):
    population_size = len(population)
    with multiprocessing.Pool(
        min(multiprocessing.cpu_count(), args["population_size"] + args["fertility"])
    ) as pool:
        if args["num_generations"] == 1:
            fitnesses = pool.map(
                functools.partial(fitness, dataset=dataset), population
            )
            rv = population[np.argmin(fitnesses)]
        else:
            try:
                for i in range(args["num_generations"] - 1):
                    population += [mutate(tpls) for tpls in crossover(population)]
                    fitnesses = pool.map(
                        functools.partial(fitness, dataset=dataset), population
                    )
                    rv = population[np.argmin(fitnesses)]
                    logging.info(
                        "generation {} has min fitness = {}".format(
                            i + 1, np.min(fitnesses)
                        )
                    )
                    selection = np.argsort(fitnesses)[:population_size]
                    population = [v for i, v in enumerate(population) if i in selection]
            except KeyboardInterrupt:
                pass
    return rv


def plot(tpls):
    sns.set_theme(style="ticks", palette="colorblind")
    x = np.linspace(0, 1, 100)
    plt.cla()
    for tpl in expand_templates(tpls):
        plt.plot(x, [tpl(v) for v in x])
    plt.title("Narrative Arc Template Curves")
    plt.xlabel("Track Number")
    plt.ylabel("Valence")
    plt.show()


def main():
    global args, training, validation, test, stats
    args = parse_args()
    if args["seed"] is not None:
        np.random.seed(args["seed"])
    (training, validation, test), stats = load_data("fma_albums_with_echonest.json")
    tpls = optimize(
        [init_templates() for _ in range(args["population_size"])], training
    )
    logging.info(
        "final templates have training fitness = {} and validation fitness = {}".format(
            fitness(tpls, training), fitness(tpls, validation)
        )
    )
    if args["outfile"] is not None:
        np.save(args["outfile"], tpls)
    if not args["no_plot"]:
        plot(tpls)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    main()
