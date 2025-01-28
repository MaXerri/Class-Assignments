Any subdirectories that exist here will be included in your ZIP file for
submission. You can use this to create your own test cases for the engine. But,
don't create any big tests here. Please keep the ZIP file size below 1000 kB,
otherwise it will be rejected by CMS. Do be sure that `make bisect` does not
need files outside this directory, and that any tests that need files outside
this directory are commented out in your test suite. Otherwise the grader will
not be able to run `make bisect`.