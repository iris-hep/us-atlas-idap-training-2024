from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

annotations = [
    "a new day, yay!",
    "US ATLAS training starts",
    "lunch",
    "afternoon low (due to hists?)",
    "1st day tutorial ends",
    "dinner at Ivar's bar!",
    "rest & nice puzzles from Jim",
    "check tomorrow's tutorials\n (Friday yay!), go to sleep!",
]
annotation_pts = [(x, np.sin(x)) for x in (6, 8, 12, 15, 17, 19, 20, 22)]
text_positions = [
    (6, -1),
    (8, 1.5),
    (12, -1.5),
    (13, 1.7),
    (16, -2),
    (17, 1.25),
    (19, 1.5),
    (19, -1),
]
SINE_FORMULA = r"$sin(x) = \sum_{n=0}^\infty\frac{(-1)^n}{(2n+1)!}x^{2n+1}$"


def sine_is_life():
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(f"THE FUNCTION OF JULY 18TH, 2024: {SINE_FORMULA}")

    ax.set_ylim(-3, 3)
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")

    x = np.linspace(6, 22, 200)
    y = np.sin(x)

    plt.xticks([6, 8, 12, 17, 22], ["6am", "8am", "12pm", "5pm", "10pm"])
    plt.yticks([-1, 1], ["not happy", "happy"])

    for text, point, text_pos in zip(annotations, annotation_pts, text_positions):
        ax.annotate(
            text.upper(),
            xy=point,
            size=10,
            xytext=text_pos,
            arrowprops=dict(arrowstyle="fancy"),
        )

    ax.plot(x, y, linewidth=3)
    fig.savefig(Path().cwd() / "sine_is_life.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    with plt.xkcd():
        sine_is_life()
