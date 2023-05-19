.PHONY: test check

build:
	dune build src

code:
	-dune build
	code .
	! dune build --watch

utop:
	OCAMLRUNPARAM=b dune utop src

test:
	OCAMLRUNPARAM=b dune exec test/main.exe

check:
	@bash check.sh

finalcheck:
	@bash check.sh final

bisect: bisect-clean
	-dune exec --instrument-with bisect_ppx --force test/main.exe
	bisect-ppx-report html

bisect-clean:
	rm -rf _coverage bisect*.coverage

zip:
	rm -f search.zip
	zip -r search.zip . -x@exclude.lst

clean: bisect-clean
	dune clean
	rm -f search.zip

doc:
	dune build @doc

opendoc: doc
	@bash opendoc.sh	
