from runtime.output_language import infer_output_language


def test_infer_zh_when_cjk_dominates() -> None:
    assert infer_output_language("组织中什么是跨部门协作？") == "zh"


def test_infer_en_when_latin_dominates() -> None:
    assert infer_output_language("How should I handle cross-team conflict in a matrix org?") == "en"


def test_infer_zh_on_empty() -> None:
    assert infer_output_language("") == "zh"
    assert infer_output_language("   ") == "zh"


def test_infer_tie_favors_zh_when_equal_counts() -> None:
    # 1 letter vs 1 CJK: latin > cjk is False -> zh
    assert infer_output_language("问a") == "zh"
