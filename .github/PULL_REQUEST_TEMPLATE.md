<!-- Please fill out the following pull request template for non-trivial changes to help us process your PR faster and more efficiently.-->

---

## Impact Analysis

| Info | Please fill out this column |
| ------ | ----------- |
| Root Cause | (Node details update was adding extra '-' in case of no variant ) |
| Internal Impact | (Update affects data contract in getEdgeNode API) |
| External Impact | (Update affects frontend-ms and deploy method in byoc-ms) |
| Ticket Link | (e.g. https://jira.devtools.intel.com/browse/DCG2-1301) |

---

## Description of contribution in a few bullet points

<!--
* I added this neat new feature
* Also fixed a typo in a parameter name in package_xxx
-->

## Tests performed

<!--
* Functionality test for getEdgeNode details
* Existing unit test was run
* Integration test for deploy endpoint
-->


### CODE MAINTAINABILITY
- [ ] Commit Message meets guidelines
- [ ] Atomic Commit -> Every commit is a single defect fix and does not mix feature addition or changes
- [ ] Added Required Tests -> Added required new tests relevant to the changes
- [ ] PR contains URL links to functional tests executed with the new tests 
- [ ] Updated Documentation as relevant to the changes
- [ ] Updated Build steps/commands changes as relevant
- [ ] PR change contains code related to security
- [ ] PR introduces breaking change. (If YES, please provide description)
- [ ] Specific instructions or information for code reviewers (If any):

### UPSTREAM EXPECTATIONS
- [ ] PR does not brake other microservices
- [ ] PR Impact is assessed
- [ ] Signed-off-by and Reviewed-by tags are correctly formatted


REVIEWER MANDATORY
------------------
### CHECKS
- [ ] Architectural and Design Fit
- [ ] Quality of code
- [ ] Commit Message meets guidelines
- [ ] PR changes adhere to industry practices and standards
- [ ] Upstream expectations are met
- [ ] Adopted domain specific coding standards
- [ ] Error and exception code paths implemented correctly
- [ ] Code reviewed for domain or language specific anti-patterns
- [ ] Code is adequately commented
- [ ] Code copyright is correct
- [ ] Tracing output are minimized and logic
- [ ] Confusing logic is explained in comments
- [ ] Commit comment can be used to design a new test case for the changes
- [ ] Test coverage shows adequate coverage with required CI tests pass on all supported platforms


## Other information

Any other relevant information.
