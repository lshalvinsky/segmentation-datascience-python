import segmentation


def test_paths_exist():
    assert segmentation.DATA_RAW.is_dir()
    assert segmentation.DATA_PROCESSED.is_dir()
