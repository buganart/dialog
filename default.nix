{ buildPythonPackage
, fetchFromGitHub
, click
, flask
, pytorch
, spacy
, tqdm
, transformers
}:

buildPythonPackage {

  pname = "dialog";
  version = "0.1.0";
  doCheck = false;

  src = ./.;

  # TODO check if still a problem with never pytorch versions
  # ERROR: Could not find a version that satisfies the requirement dataclasses (from torch->dialog==0.1.0) (from versions: none)
  pipInstallFlags = ["--no-deps"];

  propagatedBuildInputs = [
    click
    flask

    tqdm
    pytorch
    spacy

    (transformers.overridePythonAttrs (old: {
      version = "3.5.1-dev";
      src = fetchFromGitHub {
        owner = "huggingface";
        repo = "transformers";
        rev = "2819da02f7e3d0c0328daef12115d7a0cc78fc12";
        sha256 = "1xm02jmsaj9i5nj80ryml83fycr080dfw0crzzc06bzcpy5szfra";
        fetchSubmodules = true;
      };
      doCheck = false;
    }))

  ];
}
