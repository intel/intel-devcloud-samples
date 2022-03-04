# Repo Strategy

For git markdown, review [syntax guide](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

## Conventions
- Use **lowercase** wherever possible for directories
- Separate words with **hypen** `-` wherever applicable for directories

Type | Description
-----|------------
General | See [section](#General-Structure)
Branch | jiraid followed by short description e.g. `1213-repo-consistency`
Special Branch | for bugs or patches e.g. `1620-bug-short-description`, for non DEVC jira id e.g. `1620-DCG2-short-description`
Per Workload | Include `README.md` with code-block sections to build and run, adjust relative paths accordingly.

## General Structure
```
<docs>/*
<container-workloads>/<openvino version>/<type of application>/<programming language>/<application name>
```
- **openvino version:** openvino-dev-latest or openvino-lts
- **type of application:** developer-samples or oem-samples/advantech or tutorials
- **programming language:** cpp or python
- **application name:** name without language in name

**Note:** 
- Exception for oem-samples, with additional subdirectory name of the oem in `type of application`
- For any workload that doesn't fit in above structure please sync with the team to maintain consistency.

## Merge Request, Review and Branch Deletion
//TODO

## Mounted Data for Workloads
//TODO
