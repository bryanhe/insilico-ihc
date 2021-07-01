#!/usr/bin/env python3
import os
import openslide

import dstain

if __name__ == "__main__":
    pass
wsi = "data/raw"
output = "output/fig"
os.makedirs(output, exist_ok=True)


# Registration
# xy = (10000, 40000)
# size = (16384, 16384)
# downsample = 128
xy =   (54000, 20000)
size = (16384, 16384)
downsample = 32
slide = openslide.open_slide(os.path.join(wsi, "2004_38/AMY/13068_I_H.svs"))
dstain.utils.openslide.read_region_at_mag(slide, xy, 40, size, downsample=downsample).save(os.path.join(output, "he_regis.jpg"))
slide = openslide.open_slide(os.path.join(wsi, "2004_38/AMY/13081_I_aB.svs"))
dstain.utils.openslide.read_region_at_mag(slide, xy, 40, size, downsample=downsample).save(os.path.join(output, "ihc_before.jpg"))
_, transform = dstain.utils.register.register_sample(
        wsi,
        output,
        filenames=[("2004_38/AMY/13068_I_H.svs", "13068_I_H", "H"), ("2004_38/AMY/13081_I_aB.svs", "13081_I_aB", "aB")],
        window=2048,
        downsample=1,
        patches=0,
    )
dstain.utils.register.load_warped(slide, xy, size, transform[1], downsample=downsample).save(os.path.join(output, "ihc_after.jpg"))

# NM and LF
slide = openslide.open_slide(os.path.join(wsi, "2004_38/MID/13082_K_aB.svs"))
dstain.utils.openslide.read_region_at_mag(slide, (52000, 24000), 40, (2048, 2048), downsample=1).save(os.path.join(output, "lipofuscin.jpg"))
dstain.utils.openslide.read_region_at_mag(slide, (66000, 58500), 40, (2048, 2048), downsample=1).save(os.path.join(output, "neuromelanin.jpg"))

# High background Tangle
slide = openslide.open_slide(os.path.join(wsi, "2004_38/cHIP/13089_JJ_T.svs"))
dstain.utils.openslide.read_region_at_mag(slide, (63000, 26000), 40, (2048, 2048), downsample=1).save(os.path.join(output, "tangle_high.jpg"))

# Low background tangle
slide = openslide.open_slide(os.path.join(wsi, "2004_43/cHIP/13338_JJ_T.svs"))
dstain.utils.openslide.read_region_at_mag(slide, (35000, 44500), 40, (2048, 2048), downsample=1).save(os.path.join(output, "tangle_low.jpg"))

# Fold
# slide = openslide.open_slide(os.path.join(wsi, "2004_50/MF/17542_B_aB.svs"))
#slide = openslide.open_slide(os.path.join(wsi, "2005_37/HIP/18046_J_T.svs"))
#dstain.utils.openslide.read_region_at_mag(slide, (0, 0), 40, slide.dimensions, downsample=128).save(os.path.join(output, "fold.jpg"))
slide = openslide.open_slide(os.path.join(wsi, "2005_37/HIP/18046_J_T.svs"))
dstain.utils.openslide.read_region_at_mag(slide, (67000, 41000), 40, (2048, 2048), downsample=1).save(os.path.join(output, "fold.jpg"))

# 2004 43 AMY 13330 I aB  # Low background tankgle
# High background negative
# slide = openslide.open_slide(os.path.join(wsi, "2004_50/MF/17551_B_T.svs"))
# slide = openslide.open_slide(os.path.join(wsi, "2019_1819/MF/30802_B_aB.svs"))
slide = openslide.open_slide(os.path.join(wsi, "2018_48/MF/8056_B_aB.svs"))
dstain.utils.openslide.read_region_at_mag(slide, (35000, 7500), 40, (2048, 2048), downsample=1).save(os.path.join(output, "high_neg.jpg"))
# 2018_18_cHIP/13088_J_T.svs high background negative
# 2018 48 MF 8056 B aB high bg neg
# 2005 37 HIP 18046 J T fold
# 2006 06 MF 18122 B aB large fold
# 2019 0819 HIP 30732 J T ?
# 2018 48 AMY 8061 Lipo

# 2018 46 AMY 8536 I aB low background amy
slide = openslide.open_slide(os.path.join(wsi, "2018_46/AMY/8536_I_aB.svs"))
dstain.utils.openslide.read_region_at_mag(slide, (10000, 22000), 40, (2048, 2048), downsample=1).save(os.path.join(output, "amy_low.jpg"))

# 2018 46 AMY 8536 I aB high background amy
slide = openslide.open_slide(os.path.join(wsi, "2018_18/MF/7439_B_aB.svs"))
dstain.utils.openslide.read_region_at_mag(slide, (25000, 16000), 40, (2048, 2048), downsample=1).save(os.path.join(output, "amy_high.jpg"))
# dstain.utils.openslide.read_region_at_mag(slide, (0, 0), 40, slide.dimensions, downsample=128).save(os.path.join(output, "amy.jpg"))
