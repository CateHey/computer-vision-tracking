from src.common.bodily_map.findings import key_findings


def _r(f, o, ct, region, d=1.0):
    return dict(frame_idx=0, time_sec=0, focal_mouse=f, other_mouse=o, direction="", contact_type=ct,
                region=region, t=0.5, s=0, dur_sec=d)


def test_findings():
    recs = [_r("R1", "R2", "N2AG", "anogenital", 3), _r("R2", "R1", "N2B", "flank_L", 1),
            _r("R1", "R2", "N2N", "nose", 1)]
    f = {x["label"]: x for x in key_findings(recs, ["R1", "R2"])}
    assert f["Most active initiator"]["mouse"] == "R2"
    assert f["Most investigated"]["mouse"] == "R1"
    assert f["Busiest dyad"]["number"] == "5 s"
    assert f["Most rear-sniffed"]["mouse"] == "R1"
