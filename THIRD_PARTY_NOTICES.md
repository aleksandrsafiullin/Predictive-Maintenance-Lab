# Third-Party Software and Data

The Apache License 2.0 in [LICENSE](LICENSE) applies to the original source
code of Predictive Maintenance Lab. It does not replace the licenses of
third-party software, datasets, connectome data or other external resources.
Raw datasets are downloaded separately and are not distributed in this
repository. Dataset-derived material remains subject to the applicable
source terms. Dependencies installed separately retain their own copyright
and license notices; the entries below do not replace those notices or form
a complete inventory of installed Python dependencies.

## PyTorch

Copyright PyTorch contributors and the copyright holders listed in the
[upstream license](https://github.com/pytorch/pytorch/blob/v2.14.0/LICENSE).
Licensed under a BSD-style (three-clause) license.

PyTorch is installed separately as a dependency; its source and binary
distributions are not bundled in this repository.

Source: <https://github.com/pytorch/pytorch>

## three.js

Copyright © 2010-2023 three.js authors.
Licensed under the MIT License.

The repository bundles three.js 0.160.1 in
`src/pdm/visualization/component/frontend/vendor/three.min.js`.
Its original copyright and SPDX notice are retained in that file.
The complete upstream license is reproduced below.

Source: <https://github.com/mrdoob/three.js>
License: <https://github.com/mrdoob/three.js/blob/r160/LICENSE>

```text
The MIT License

Copyright © 2010-2023 three.js authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## Streamlit custom-component bridge

Copyright Streamlit Inc. (2018-2022) / Snowflake Inc. (2022), as recorded in
`src/pdm/visualization/component/frontend/vendor/VENDOR.txt`.
Licensed under the Apache License 2.0.

`src/pdm/visualization/component/frontend/vendor/streamlit-component-lib.js`
is a local browser shim of the Streamlit custom-component iframe protocol,
as indicated in its header. It is not a bundled copy of the published
ESM/React npm package. The original attribution and local-modification
description are retained.

Source: <https://github.com/streamlit/streamlit>
License: <https://github.com/streamlit/streamlit/blob/develop/LICENSE>
The complete Apache License 2.0 is included in [LICENSE](LICENSE).

## MaleCNS Connectome

Male CNS Connectome v1.0.

MaleCNS connectome data © FlyEM / HHMI Janelia, University of Cambridge
(Department of Zoology), MRC Laboratory of Molecular Biology and Google Research.

Dataset licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
as stated by the [official download page](https://male-cns.janelia.org/download/).

Source: <https://male-cns.janelia.org/>

The original data files are downloaded separately. Predictive Maintenance
Lab selects annotated neurons, aggregates connections, rescales connection
weights and generates computational reservoir states and visualizations.
These are project transformations, not recordings of biological activity
or outputs endorsed by the dataset contributors. See
[Full MaleCNS computation](docs/full_cns_signal.md) and
[connectome processing](docs/fly_connectome.md) for the transformations.

The MaleCNS dataset and dataset-derived material are not covered by the
Predictive Maintenance Lab Apache-2.0 license. They remain subject to their
original CC BY 4.0 license and attribution requirements.

## XJTU-SY Bearing Dataset

Provided by the Institute of Design Science and Basic Component at
Xi'an Jiaotong University and Changxing Sumyoung Technology Co., Ltd.
Original dataset copyright and usage terms remain with the dataset authors.
The dataset is downloaded separately and is not distributed as part of
this repository or relicensed under Apache-2.0.

Source and usage terms: <https://biaowang.tech/xjtu-sy-bearing-datasets/>

The authors request that publications using the dataset cite:
Biao Wang, Yaguo Lei, Naipeng Li and Ningbo Li,
"A Hybrid Prognostics Approach for Estimating Remaining Useful Life of
Rolling Element Bearings," IEEE Transactions on Reliability,
vol. 69, no. 1, pp. 401-412, 2020.

## HSE Predictive Maintenance Dataset

Preventive to Predictive Maintenance dataset, published by Prognostics HSE.
Original dataset copyright and licensing remain with the dataset authors.
The dataset is downloaded separately and is not distributed as part of
this repository or relicensed under Apache-2.0.

Source and dataset terms:
<https://www.kaggle.com/datasets/prognosticshse/preventive-to-predicitve-maintenance>

Associated publication: S. Hagmeyer, F. Mauthe and P. Zeiler,
"Creation of Publicly Available Data Sets for Prognostics and Diagnostics
Addressing Data Scenarios Relevant to Industrial Applications,"
International Journal of Prognostics and Health Management, 2021.
<https://papers.phmsociety.org/index.php/ijphm/article/view/3087>

Consult the dataset distribution and its authors' terms for permitted use
and redistribution; the paper's license does not replace the dataset terms.
