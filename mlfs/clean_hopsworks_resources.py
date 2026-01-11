import argparse
import hopsworks


def delete_model_versions(mr, model_name: str, dry_run: bool):
    try:
        models = mr.get_models(name=model_name)  # all versions for this name
    except Exception:
        return

    for m in models:
        print(f"  - Model: {m.name} v{m.version}")
        if not dry_run:
            try:
                m.delete()
            except Exception as e:
                print(f"    ! Failed deleting model {m.name} v{m.version}: {e}")


def delete_feature_view_versions(fs, fv_name: str, dry_run: bool):
    try:
        fvs = fs.get_feature_views(name=fv_name)  # all versions for this name
    except Exception:
        return

    for fv in fvs:
        print(f"  - Feature view: {fv.name} v{fv.version}")
        if not dry_run:
            try:
                fv.delete()
            except Exception as e:
                print(f"    ! Failed deleting feature view {fv.name} v{fv.version}: {e}")


def delete_feature_group_versions(fs, fg_name: str, dry_run: bool):
    try:
        fgs = fs.get_feature_groups(name=fg_name)  # all versions for this name
    except Exception:
        return

    for fg in fgs:
        print(f"  - Feature group: {fg.name} v{fg.version}")
        if not dry_run:
            try:
                fg.delete()
            except Exception as e:
                print(f"    ! Failed deleting feature group {fg.name} v{fg.version}: {e}")


def delete_deployment_if_exists(ms, deployment_name: str, dry_run: bool):
    try:
        dep = ms.get_deployment(name=deployment_name)
    except Exception:
        return

    print(f"  - Deployment: {dep.name}")
    if not dry_run:
        try:
            dep.stop()
        except Exception:
            pass
        try:
            dep.delete()
        except Exception as e:
            print(f"    ! Failed deleting deployment {dep.name}: {e}")


def try_list_all_names(obj, list_method_candidates):
    """
    Best-effort: Hopsworks API differs by version. We try a few method names.
    Returns list of objects or [].
    """
    for m in list_method_candidates:
        if hasattr(obj, m):
            try:
                res = getattr(obj, m)()
                return res if res is not None else []
            except Exception:
                continue
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--prefix",
        default="mcphases_",
        help="Delete only resources whose names start with this prefix (default: mcphases_).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted, but do not delete anything.",
    )
    ap.add_argument(
        "--delete-kafka",
        action="store_true",
        help="Also delete Kafka topics that start with the prefix (ONLY if you used online feature groups).",
    )
    args = ap.parse_args()

    prefix = args.prefix
    dry_run = args.dry_run

    print(f"Cleaning Hopsworks resources with prefix: '{prefix}' (dry_run={dry_run})")

    project = hopsworks.login(engine="python")
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    ms = project.get_model_serving()
    kafka_api = project.get_kafka_api()

    # 1) Collect candidate names (best-effort listing)
    # Some Hopsworks versions allow listing all; if not, we fall back to “known names” below.
    all_fgs = try_list_all_names(fs, ["get_feature_groups"])
    all_fvs = try_list_all_names(fs, ["get_feature_views"])
    all_models = try_list_all_names(mr, ["get_models"])
    all_deps = try_list_all_names(ms, ["get_deployments"])

    fg_names = sorted({fg.name for fg in all_fgs if getattr(fg, "name", "").startswith(prefix)})
    fv_names = sorted({fv.name for fv in all_fvs if getattr(fv, "name", "").startswith(prefix)})
    model_names = sorted({m.name for m in all_models if getattr(m, "name", "").startswith(prefix)})
    dep_names = sorted({d.name for d in all_deps if getattr(d, "name", "").startswith(prefix)})

    # Fallback if listing isn’t supported (very common):
    # Put your actual names here once you create them.
    # This makes the script still usable even if list APIs aren’t available.
    if not fg_names and not fv_names and not model_names and not dep_names:
        print("Could not list resources via API. Using fallback known-name patterns.")
        # Add to these as you create them in your project:
        fg_names = [
            "mcphases_daily_fg",
            "mcphases_predictions_fg",
        ]
        fv_names = [
            "mcphases_daily_fv",
        ]
        model_names = [
            "mcphases_mood_today_extratrees",
            "mcphases_fatigue_today_extratrees",
            "mcphases_mood_tomorrow_extratrees",
            "mcphases_fatigue_tomorrow_extratrees",
        ]
        dep_names = []  # only if you actually deployed something

    print("\nPlanned deletions (in safe order):")
    print("Deployments:", dep_names or "none")
    print("Models:", model_names or "none")
    print("Feature views:", fv_names or "none")
    print("Feature groups:", fg_names or "none")

    # 2) Delete in safe dependency order:
    # deployments -> models -> feature views -> feature groups
    if dep_names:
        print("\nDeleting deployments...")
        for name in dep_names:
            delete_deployment_if_exists(ms, name, dry_run)

    if model_names:
        print("\nDeleting models (all versions)...")
        for name in model_names:
            delete_model_versions(mr, name, dry_run)

    if fv_names:
        print("\nDeleting feature views (all versions)...")
        for name in fv_names:
            delete_feature_view_versions(fs, name, dry_run)

    if fg_names:
        print("\nDeleting feature groups (all versions)...")
        for name in fg_names:
            delete_feature_group_versions(fs, name, dry_run)

    # 3) Optional Kafka cleanup (ONLY if you used online feature groups / Kafka topics)
    if args.delete_kafka:
        print("\nDeleting Kafka topics (prefix match)...")
        try:
            topics = kafka_api.get_topics()
            for t in topics:
                if t.name.startswith(prefix):
                    print(f"  - Kafka topic: {t.name}")
                    if not dry_run:
                        try:
                            t.delete()
                        except Exception as e:
                            print(f"    ! Failed deleting Kafka topic {t.name}: {e}")
        except Exception as e:
            print(f"Kafka cleanup skipped (could not list topics): {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()