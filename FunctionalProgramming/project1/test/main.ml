open OUnit2
open Enigma

(** [index_test name input expected_output] constructs an OUnit test named
    [name] that asserts the quality of [expected_output] with [index input]. *)
let index_test (name : string) (input : char) (expected_output : int) : test =
  name >:: fun _ ->
  (* the [printer] tells OUnit how to convert the output to a string *)
  assert_equal expected_output (index input) ~printer:string_of_int

(* You will find it helpful to write functions like [index_test] for each of the
   other functions you are testing. They will keep your lists of tests below
   very readable, and will also help you to avoid repeating code. You will also
   find it helpful to create [~printer] functions for the data types in use. *)

let index_tests =
  [
    index_test "index of A is 0" 'A' 0;
    index_test "index of B is 1" 'B' 1;
    index_test "index of Z is 90" 'Z' 25;
    index_test "index of M is 77" 'M' 12;
  ]

(** [map_rl_test name input1 input2 input3 expected_output] constructs an OUnit
    test named [name] that asserts the quality of [expected_output] with
    [map_r_to_l input1 input2 input3]. *)
let map_r1_test (name : string) (input1 : string) (input2 : char) (input3 : int)
    (expected_output : int) : test =
  name >:: fun _ ->
  assert_equal expected_output
    (map_r_to_l input1 input2 input3)
    ~printer:string_of_int

let map_rl_tests =
  [
    map_r1_test "left map basic1" "ABCDEFGHIJKLMNOPQRSTUVWXYZ" 'A' 0 0;
    map_r1_test "left map basic2" "ABCDEFGHIJKLMNOPQRSTUVWXYZ" 'A' 2 2;
    map_r1_test "left map basic3" "BDFHJLCPRTXVZNYEIWGAKMUSQO" 'X' 10 18;
    map_r1_test "left map basic4" "EKMFLGDQVZNTOWYHXUSPAIBRCJ" 'G' 25 0;
    map_r1_test "left map basic5" "EKMFLGDQVZNTOWYHXUSPAIBRCJ" 'Z' 25 3;
    map_r1_test "left map basic5" "EKMFLGDQVZNTOWYHXUSPAIBRCJ" 'Z' 25 3;
    map_r1_test "left map basic6" "BDFHJLCPRTXVZNYEIWGAKMUSQO" 'O' 14 17;
    map_r1_test "left map basic7" "IMETCGFRAYSQBZXWLHKDVUPOJN" 'B' 3 1;
  ]

(** [map_lr_test name input1 input2 input3 expected_output] constructs an OUnit
    test named [name] that asserts the quality of [expected_output] with
    [map_l_to_r input1 input2 input3]. *)
let map_lr_test (name : string) (input1 : string) (input2 : char) (input3 : int)
    (expected_output : int) : test =
  name >:: fun _ ->
  assert_equal expected_output
    (map_l_to_r input1 input2 input3)
    ~printer:string_of_int

let map_lr_tests =
  [
    map_lr_test "right map basic1" "EKMFLGDQVZNTOWYHXUSPAIBRCJ" 'F' 10 14;
    map_lr_test "right map basic2" "AJDKSIRUXBLHWTMCQGZNPYFVOE" 'A' 0 0;
    map_lr_test "right map basic3" "EKMFLGDQVZNTOWYHXUSPAIBRCJ" 'Z' 25 15;
    map_lr_test "right map basic4" "BDFHJLCPRTXVZNYEIWGAKMUSQO" 'G' 15 5;
    map_lr_test "right map basic5" "BDFHJLCPRTXVZNYEIWGAKMUSQO" 'Z' 0 13;
    map_lr_test "right map basic6" "IMETCGFRAYSQBZXWLHKDVUPOJN" 'B' 1 3;
  ]

(** [map_refl_test name input1 input2 expected_output] constructs an OUnit test
    named [name] that asserts the quality of [expected_output] with
    [map_refl input1 input2]. *)
let map_refl_test (name : string) (input1 : string) (input2 : int)
    (expected_output : int) : test =
  name >:: fun _ ->
  assert_equal expected_output (map_refl input1 input2) ~printer:string_of_int

let map_refl_tests =
  [
    map_refl_test "refl basic1" "YRUHQSLDPXNGOKMIEBFZCWVJAT" 0 24;
    map_refl_test "refl basic2" "YRUHQSLDPXNGOKMIEBFZCWVJAT" 24 0;
    map_refl_test "refl basic3" "FVPJIAOYEDRZXWGCTKUQSBNMHL" 25 11;
    map_refl_test "refl basic4" "FVPJIAOYEDRZXWGCTKUQSBNMHL" 11 25;
    map_refl_test "refl basic5" "ABCDEFGHIJKLMNOPQRSTUVWXYZ" 1 1;
    map_refl_test "refl basic6" "IMETCGFRAYSQBZXWLHKDVUPOJN" 2 4;
    map_refl_test "refl basic7" "IMETCGFRAYSQBZXWLHKDVUPOJN" 2 4;
    map_refl_test "refl basic8" "CIAGSNDRBYTPZFULVHEKOQXWJM" 23 22;
    map_refl_test "refl basic9" "CIAGSNDRBYTPZFULVHEKOQXWJM" 22 23;
  ]

(** [map_plug_test name input1 input2 expected_output] constructs an OUnit test
    named [name] that asserts the quality of [expected_output] with
    [map_plug input1 input2]. *)
let map_plug_test (name : string) (input1 : (char * char) list) (input2 : char)
    (expected_output : char) : test =
  name >:: fun _ ->
  assert_equal expected_output (map_plug input1 input2) ~printer:Char.escaped

let map_plug_tests =
  [
    map_plug_test "test1" [] 'A' 'A';
    map_plug_test "test2" [ ('A', 'B') ] 'A' 'B';
    map_plug_test "test3" [ ('A', 'B') ] 'B' 'A';
    map_plug_test "test4" [ ('A', 'B'); ('D', 'C') ] 'D' 'C';
    map_plug_test "test5" [ ('A', 'B'); ('D', 'C') ] 'E' 'E';
    map_plug_test "test6" [ ('B', 'A'); ('C', 'D') ] 'D' 'C';
    map_plug_test "test7"
      [
        ('A', 'B');
        ('C', 'D');
        ('E', 'F');
        ('G', 'H');
        ('I', 'J');
        ('K', 'L');
        ('M', 'N');
        ('O', 'P');
        ('Q', 'R');
        ('S', 'T');
        ('U', 'V');
        ('W', 'X');
        ('Y', 'Z');
      ]
      'Z' 'Y';
  ]

(** [cipher_char_test name input1 input2 expected_output] constructs an OUnit
    test named [name] that asserts the quality of [expected_output] with
    [map_plug input1 input2]. *)
let cipher_char_test (name : string) (input1 : config) (input2 : char)
    (expected_output : char) : test =
  name >:: fun _ ->
  assert_equal expected_output (cipher_char input1 input2) ~printer:Char.escaped

let cipher_char_tests =
  [
    cipher_char_test "one rotor"
      { refl = "FVPJIAOYEDRZXWGCTKUQSBNMHL"; rotors = []; plugboard = [] }
      'A' 'F';
    cipher_char_test "no_plugboard"
      {
        refl = "YRUHQSLDPXNGOKMIEBFZCWVJAT";
        rotors =
          [
            {
              rotor = { wiring = "EKMFLGDQVZNTOWYHXUSPAIBRCJ"; turnover = 'A' };
              top_letter = 'A';
            };
            {
              rotor = { wiring = "AJDKSIRUXBLHWTMCQGZNPYFVOE"; turnover = 'A' };
              top_letter = 'A';
            };
            {
              rotor = { wiring = "BDFHJLCPRTXVZNYEIWGAKMUSQO"; turnover = 'A' };
              top_letter = 'A';
            };
          ];
        plugboard = [];
      }
      'G' 'P';
    cipher_char_test "complete"
      {
        refl = "YRUHQSLDPXNGOKMIEBFZCWVJAT";
        rotors =
          [
            {
              rotor = { wiring = "EKMFLGDQVZNTOWYHXUSPAIBRCJ"; turnover = 'A' };
              top_letter = 'C';
            };
            {
              rotor = { wiring = "AJDKSIRUXBLHWTMCQGZNPYFVOE"; turnover = 'A' };
              top_letter = 'H';
            };
            {
              rotor = { wiring = "BDFHJLCPRTXVZNYEIWGAKMUSQO"; turnover = 'A' };
              top_letter = 'Z';
            };
          ];
        plugboard = [ ('B', 'F'); ('E', 'Q'); ('X', 'V') ];
      }
      'B' 'Q';
    cipher_char_test "empty_rotor_list"
      {
        refl = "IMETCGFRAYSQBZXWLHKDVUPOJN";
        rotors = [];
        plugboard = [ ('V', 'Z'); ('E', 'Q'); ('X', 'V') ];
      }
      'V' 'N';
    cipher_char_test "one rotor"
      {
        refl = "IMETCGFRAYSQBZXWLHKDVUPOJN";
        rotors =
          [
            {
              rotor = { wiring = "IMETCGFRAYSQBZXWLHKDVUPOJN"; turnover = 'A' };
              top_letter = 'C';
            };
          ];
        plugboard = [];
      }
      'E' 'S';
  ]

(** [push_test name input1 expected_output] constructs an OUnit test named
    [name] that asserts the quality of [expected_output] with
    [map_plug input1 input2]. *)
let step_test (name : string) (input1 : config) (expected_output : config) :
    test =
  name >:: fun _ -> assert_equal expected_output (step input1)

let step_tests =
  [
    step_test "complete"
      {
        refl = "YRUHQSLDPXNGOKMIEBFZCWVJAT";
        rotors =
          [
            {
              rotor = { wiring = "EKMFLGDQVZNTOWYHXUSPAIBRCJ"; turnover = 'A' };
              top_letter = 'C';
            };
            {
              rotor = { wiring = "AJDKSIRUXBLHWTMCQGZNPYFVOE"; turnover = 'A' };
              top_letter = 'H';
            };
            {
              rotor = { wiring = "BDFHJLCPRTXVZNYEIWGAKMUSQO"; turnover = 'A' };
              top_letter = 'Z';
            };
          ];
        plugboard = [ ('B', 'F'); ('E', 'Q'); ('X', 'V') ];
      }
      {
        refl = "YRUHQSLDPXNGOKMIEBFZCWVJAT";
        rotors =
          [
            {
              rotor = { wiring = "EKMFLGDQVZNTOWYHXUSPAIBRCJ"; turnover = 'A' };
              top_letter = 'C';
            };
            {
              rotor = { wiring = "AJDKSIRUXBLHWTMCQGZNPYFVOE"; turnover = 'A' };
              top_letter = 'H';
            };
            {
              rotor = { wiring = "BDFHJLCPRTXVZNYEIWGAKMUSQO"; turnover = 'A' };
              top_letter = 'Z';
            };
          ];
        plugboard = [ ('B', 'F'); ('E', 'Q'); ('X', 'V') ];
      };
  ]

let cipher_tests = [ (* TODO: add your tests here *) ]

let tests =
  "test suite for A1"
  >::: List.flatten
         [
           index_tests;
           map_rl_tests;
           map_lr_tests;
           map_refl_tests;
           map_plug_tests;
           cipher_char_tests;
           step_tests;
           cipher_tests;
         ]

let _ = run_test_tt_main tests
