import sys
from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def main():
    xml_paths = sorted(MODELS_DIR.glob("*.xml"))

    has_error = False

    print("=" * 80)
    print(f"Checking MuJoCo models in: {MODELS_DIR}")
    print("=" * 80)

    for xml_path in xml_paths:
        try:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            print(
                f"{xml_path.name:20s} | "
                f"model={xml_path.stem:18s} | "
                f"nq={model.nq:2d} | nv={model.nv:2d} | nu={model.nu:2d} | "
                f"nbody={model.nbody:2d} | ngeom={model.ngeom:2d} | "
                f"nsensor={model.nsensor:2d} | nsensordata={model.nsensordata:2d}"
            )
        except Exception as exc:
            has_error = True
            print(f"{xml_path.name:20s} | LOAD FAILED | {exc}")

    if has_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
