"""Write tiny, valid mzML files for tests (optionally with ion mobility and
MS2 precursor metadata), so the loader is exercised through pymzml."""

import base64
import zlib

import numpy as np

_ARRAY_CV = {
    "mz": ("MS:1000514", "m/z array"),
    "intensity": ("MS:1000515", "intensity array"),
    "mobility": ("MS:1003006", "mean inverse reduced ion mobility array"),
}


def _array_xml(kind, values):
    raw = np.asarray(values, dtype="<f8").tobytes()
    data = base64.b64encode(zlib.compress(raw)).decode()
    acc, name = _ARRAY_CV[kind]
    unit = (
        'unitCvRef="MS" unitAccession="MS:1002814" '
        'unitName="volt-second per square centimeter"'
        if kind == "mobility"
        else (
            'unitCvRef="MS" unitAccession="MS:1000040" unitName="m/z"'
            if kind == "mz"
            else 'unitCvRef="MS" unitAccession="MS:1000131" '
            'unitName="number of detector counts"'
        )
    )
    return (
        f'<binaryDataArray encodedLength="{len(data)}">'
        '<cvParam cvRef="MS" accession="MS:1000523" name="64-bit float" value=""/>'
        '<cvParam cvRef="MS" accession="MS:1000574" name="zlib compression" value=""/>'
        f'<cvParam cvRef="MS" accession="{acc}" name="{name}" value="" {unit}/>'
        f"<binary>{data}</binary></binaryDataArray>"
    )


def spectrum(
    mz,
    intensity,
    rt_seconds,
    ms_level=1,
    mobility=None,
    precursor=None,
    charge=None,
    window=None,
    scan_mobility=None,
):
    return dict(
        mz=mz,
        intensity=intensity,
        rt=rt_seconds,
        ms_level=ms_level,
        mobility=mobility,
        precursor=precursor,
        charge=charge,
        window=window,
        scan_mobility=scan_mobility,
    )


def write_mzml(path, spectra):
    out = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<mzML xmlns="http://psi.hupo.org/ms/mzml" version="1.1.0">',
        '<cvList count="2"><cv id="MS" fullName="PSI-MS" version="4.1.0" '
        'URI="https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"/>'
        '<cv id="UO" fullName="Unit Ontology" version="1" '
        'URI="https://raw.githubusercontent.com/bio-ontology-research-group/unit-ontology/master/unit.obo"/></cvList>',
        '<run id="test">',
        f'<spectrumList count="{len(spectra)}" defaultDataProcessingRef="dp">',
    ]
    for i, s in enumerate(spectra):
        n = len(s["mz"])
        out.append(
            f'<spectrum index="{i}" id="scan={i + 1}" '
            f'defaultArrayLength="{n}">'
        )
        out.append(
            '<cvParam cvRef="MS" accession="MS:1000511" '
            f'name="ms level" value="{s["ms_level"]}"/>'
        )
        out.append(
            '<cvParam cvRef="MS" accession="MS:1000127" '
            'name="centroid spectrum" value=""/>'
        )
        out.append('<scanList count="1"><scan>')
        out.append(
            '<cvParam cvRef="MS" accession="MS:1000016" '
            f'name="scan start time" value="{s["rt"]}" unitCvRef="UO" '
            'unitAccession="UO:0000010" unitName="second"/>'
        )
        if s["scan_mobility"] is not None:
            out.append(
                '<cvParam cvRef="MS" accession="MS:1002815" '
                'name="inverse reduced ion mobility" '
                f'value="{s["scan_mobility"]}" unitCvRef="MS" '
                'unitAccession="MS:1002814" '
                'unitName="volt-second per square centimeter"/>'
            )
        out.append("</scan></scanList>")
        if s["precursor"] is not None:
            out.append('<precursorList count="1"><precursor>')
            if s["window"] is not None:
                target, off = s["window"]
                out.append(
                    "<isolationWindow>"
                    '<cvParam cvRef="MS" accession="MS:1000827" '
                    f'name="isolation window target m/z" value="{target}"/>'
                    '<cvParam cvRef="MS" accession="MS:1000828" '
                    f'name="isolation window lower offset" value="{off}"/>'
                    '<cvParam cvRef="MS" accession="MS:1000829" '
                    f'name="isolation window upper offset" value="{off}"/>'
                    "</isolationWindow>"
                )
            out.append(
                '<selectedIonList count="1"><selectedIon>'
                '<cvParam cvRef="MS" accession="MS:1000744" '
                f'name="selected ion m/z" value="{s["precursor"]}"/>'
            )
            if s["charge"] is not None:
                out.append(
                    '<cvParam cvRef="MS" accession="MS:1000041" '
                    f'name="charge state" value="{s["charge"]}"/>'
                )
            out.append(
                "</selectedIon></selectedIonList></precursor>"
                "</precursorList>"
            )
        arrays = [("mz", s["mz"]), ("intensity", s["intensity"])]
        if s["mobility"] is not None:
            arrays.append(("mobility", s["mobility"]))
        out.append(f'<binaryDataArrayList count="{len(arrays)}">')
        out.extend(_array_xml(kind, vals) for kind, vals in arrays)
        out.append("</binaryDataArrayList></spectrum>")
    out.append("</spectrumList></run></mzML>")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out))
    return str(path)
