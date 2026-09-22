import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from aiweather.forecast import open_forecast
from aiweather.tracking import build_native_tc_tracks


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot AIWeather multi-system native tropical cyclone "
            "tracks from one operational forecast cycle."
        )
    )
    parser.add_argument(
        "--init",
        required=True,
        help="Initialization time as YYYYMMDDTHHMMSS.",
    )
    parser.add_argument(
        "--western-name",
        default="Western",
        help="Name of the western tropical cyclone.",
    )
    parser.add_argument(
        "--eastern-name",
        default="Eastern",
        help="Name of the eastern tropical cyclone.",
    )
    parser.add_argument(
        "--maximum-translation-speed-mps",
        type=float,
        default=20.0,
        help=(
            "Maximum allowed translation speed in m/s. "
            "Default: 20."
        ),
    )
    return parser.parse_args()


def lon180(lon):
    return lon - 360.0 if lon > 180.0 else lon


def main():
    args = parse_args()

    init = args.init
    western_name = args.western_name
    eastern_name = args.eastern_name

    forecasts = {
        "GraphCast": Path(
            f"outputs/graphcast/{init}/forecast.zarr"
        ),
        "AIFS2": Path(
            f"outputs/aifs2/{init}/forecast_azure.zarr"
        ),
        "Pangu3": Path(
            f"outputs/pangu3/{init}/forecast.zarr"
        ),
        "Pangu6": Path(
            f"outputs/pangu6/{init}/forecast.zarr"
        ),
    }

    systems = {
        western_name: [],
        eastern_name: [],
    }

    for model, path in forecasts.items():
        forecast = open_forecast(path)

        results = build_native_tc_tracks(
            forecast,
            lat_min=5.0,
            lat_max=35.0,
            lon_min=-140.0,
            lon_max=-90.0,
            maximum_translation_speed_mps=(
                args.maximum_translation_speed_mps
            ),
        )

        for genesis, records in results:
            glon = lon180(genesis.longitude)

            # Geographic classification based on native
            # detection location.
            if glon < -115.0:
                group = western_name
            else:
                group = eastern_name

            systems[group].append(
                (model, genesis, records)
            )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        constrained_layout=True,
    )

    domains = {
        western_name: (-135, -115, 10, 35),
        eastern_name: (-125, -95, 10, 35),
    }

    for ax, group in zip(
        axes,
        (western_name, eastern_name),
    ):
        for model, genesis, records in systems[group]:
            lons = [
                lon180(record.longitude)
                for record in records
            ]
            lats = [
                record.latitude
                for record in records
            ]

            line, = ax.plot(
                lons,
                lats,
                marker=".",
                linewidth=1.7,
                label=model,
            )

            # Native-detection marker.
            ax.scatter(
                lon180(genesis.longitude),
                genesis.latitude,
                marker="*",
                s=150,
                color=line.get_color(),
                zorder=5,
            )

            # Mark every 24 forecast hours.
            for record in records:
                if record.lead_time_hours % 24 == 0:
                    ax.scatter(
                        lon180(record.longitude),
                        record.latitude,
                        s=25,
                        color=line.get_color(),
                        zorder=4,
                    )

        xmin, xmax, ymin, ymax = domains[group]

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.3)

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")

        if group == western_name:
            title = f"{western_name} — western system"
        else:
            title = f"{eastern_name} — eastern system"

        ax.set_title(title)
        ax.legend()

    init_dt = datetime.strptime(
        init,
        "%Y%m%dT%H%M%S",
    )

    fig.suptitle(
        "AIWeather operational tropical cyclone forecast — "
        f"{eastern_name} and {western_name}\n"
        f"Initialization: "
        f"{init_dt:%Y-%m-%d %H} UTC",
        fontsize=14,
    )

    storm_slug = (
        f"{eastern_name}_{western_name}"
        .lower()
        .replace(" ", "_")
    )

    output = Path(
        "outputs/verification"
    ) / f"{storm_slug}_{init}.png"

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(output)


if __name__ == "__main__":
    main()
