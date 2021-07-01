# regions = ("AMY", "cHIP", "HIP", "MID", "MF", "PVC", "IPL", "Striatum", "Cerebellum")
regions = ("AMY", "cHIP", "HIP", "MID", "MF", "PVC", "IPL", "Striatum")

stains = ("aB", "T")
annotations = (
    "Unstained",
    # "Amyloid Deposit",
    "pTau Tangle",
    # "pTau Thread",
    # "Neuritic Plaque",
    # "Pick Body",
    # "Neuromelanin",
    # "Background",
    "Artifact",
)

annotations_in_stain = {
    "T": (
        # "Unstained",
        # "pTau Tangle",
        "pTau Thread",
        # "Neuritic Plaque",
        # "Pick Body",
        # "Background",
        "Artifact",
    )
    # TODO: incomplete
}

relevant_for_stain = {
    "T": (
        ("pTau Tangle", 0.005),
        # ("pTau Tangle", 0.02),
        # ("pTau Thread", 0.2),
    )
    # TODO: incomplete
}
