open OUnit2
open Game
open Adventure
open Command
open State

(********************************************************************
   Here are some helper functions for your testing of set-like lists.
 ********************************************************************)

(** [cmp_set_like_lists lst1 lst2] compares two lists to see whether they are
    equivalent set-like lists. That means checking two things. First, they must
    both be "set-like", meaning that they do not contain any duplicates. Second,
    they must contain the same elements, though not necessarily in the same
    order. *)
let cmp_set_like_lists lst1 lst2 =
  let uniq1 = List.sort_uniq compare lst1 in
  let uniq2 = List.sort_uniq compare lst2 in
  List.length lst1 = List.length uniq1
  && List.length lst2 = List.length uniq2
  && uniq1 = uniq2

(** [pp_string s] pretty-prints string [s]. *)
let pp_string s = "\"" ^ s ^ "\""

(** [pp_list pp_elt lst] pretty-prints list [lst], using [pp_elt] to
    pretty-print each element of [lst]. *)
let pp_list pp_elt lst =
  let pp_elts lst =
    let rec loop n acc = function
      | [] -> acc
      | [ h ] -> acc ^ pp_elt h
      | h1 :: (h2 :: t as t') ->
          if n = 100 then acc ^ "..." (* stop printing long list *)
          else loop (n + 1) (acc ^ pp_elt h1 ^ "; ") t'
    in
    loop 0 "" lst
  in
  "[" ^ pp_elts lst ^ "]"

(* These tests demonstrate how to use [cmp_set_like_lists] and [pp_list] to get
   helpful output from OUnit. *)
let cmp_demo =
  [
    ( "order is irrelevant" >:: fun _ ->
      assert_equal ~cmp:cmp_set_like_lists ~printer:(pp_list pp_string)
        [ "foo"; "bar" ] [ "bar"; "foo" ] )
    (* Uncomment this test to see what happens when a test case fails.
       "duplicates not allowed" >:: (fun _ -> assert_equal
       ~cmp:cmp_set_like_lists ~printer:(pp_list pp_string) ["foo"; "foo"]
       ["foo"]); *);
  ]

(********************************************************************
   End helper functions.
 ********************************************************************)

(* You are welcome to add strings containing JSON here, and use them as the
   basis for unit tests. You can also use the JSON files in the data directory
   as tests. And you can add JSON files in this directory and use them, too. *)

(* Here is an example of how to load files from the data directory: *)
let data_dir_prefix = "data" ^ Filename.dir_sep
let lonely = Yojson.Basic.from_file (data_dir_prefix ^ "lonely_room.json")
let ho = Yojson.Basic.from_file (data_dir_prefix ^ "ho_plaza.json")

(* You should not be testing any helper functions here. Test only the functions
   exposed in the [.mli] files. Do not expose your helper functions. See the
   handout for an explanation. *)

(** [start_room_test name input expected_output] constructs an OUnit test named
    [name] that asserts the quality of [expected_output] with
    [start_room input]. *)
let start_room_test (name : string) (input : Adventure.t)
    (expected_output : string) : test =
  name >:: fun _ ->
  assert_equal expected_output (start_room input) ~printer:pp_string

(** [room_ids_test name input expected_output] constructs an OUnit test named
    [name] that asserts the quality of [expected_output] with
    [start_room input]. *)
let room_ids_test (name : string) (input : Adventure.t)
    (expected_output : string list) : test =
  name >:: fun _ ->
  assert_equal ~cmp:cmp_set_like_lists ~printer:(pp_list pp_string)
    expected_output (room_ids input)

(** [description_test name input expected_output] constructs an OUnit test named
    [name] that asserts the quality of [expected_output] with
    [start_room input1 input2]. *)
let description_test (name : string) (input1 : Adventure.t) (input2 : string)
    (expected_output : string) : test =
  name >:: fun _ ->
  assert_equal expected_output (description input1 input2) ~printer:pp_string

(** [exits_test name input expected_output] constructs an OUnit test named
    [name] that asserts the quality of [expected_output] with
    [start_room input1 input2]. *)
let exits_test (name : string) (input1 : Adventure.t) (input2 : string)
    (expected_output : string list) : test =
  name >:: fun _ ->
  assert_equal ~cmp:cmp_set_like_lists ~printer:(pp_list pp_string)
    expected_output (exits input1 input2)

(** [next_room_test name input expected_output] constructs an OUnit test named
    [name] that asserts the quality of [expected_output] with
    [start_room input1 input2 input3]. *)
let next_room_test (name : string) (input1 : Adventure.t) (input2 : string)
    (input3 : string) (expected_output : string) : test =
  name >:: fun _ ->
  assert_equal expected_output
    (next_room input1 input2 input3)
    ~printer:pp_string

(** [next_rooms_test name input expected_output] constructs an OUnit test named
    [name] that asserts the quality of [expected_output] with
    [start_room input1 input2]. *)
let next_rooms_test (name : string) (input1 : Adventure.t) (input2 : string)
    (expected_output : string list) : test =
  name >:: fun _ ->
  assert_equal ~cmp:cmp_set_like_lists ~printer:(pp_list pp_string)
    expected_output (next_rooms input1 input2)

let adventure_tests =
  [
    (*Tests for start_room*)
    start_room_test "lonely start room" (from_json lonely) "the room";
    start_room_test "Ho Plaza Start Room" (from_json ho) "ho plaza";
    (*Tests for room_ids*)
    room_ids_test "lonely start room" (from_json lonely) [ "the room" ];
    room_ids_test "ho plaza room ids" (from_json ho)
      [ "ho plaza"; "health"; "tower"; "nirvana" ];
    room_ids_test "ho plaza testids with list in different order" (from_json ho)
      [ "ho plaza"; "health"; "nirvana"; "tower" ];
    (*tests for desciption*)
    description_test "lonely description for 'the room' " (from_json lonely)
      "the room" "A very lonely room.";
    ( "description exception" >:: fun _ ->
      assert_raises (UnknownRoom "This room doesn't exist") (fun () ->
          description (from_json lonely) "the r") );
    description_test "ho plaza description test with health room" (from_json ho)
      "health"
      "You are at the entrance to Cornell Health. A sign advertises free flu \
       shots. You briefly wonder how long it would take to get an appointment. \
       Ho Plaza is to the northeast.";
    description_test "ho plaza description for nirvana" (from_json ho) "nirvana"
      "You have reached a higher level of existence.  There are no more words.";
    (*Tests for exits*)
    ( "exits exception for lonely" >:: fun _ ->
      assert_raises (UnknownRoom "This room doesn't exist") (fun () ->
          exits (from_json lonely) "hi") );
    exits_test "exits ho plaza test w/o exits" (from_json ho) "nirvana" [];
    exits_test "exits ho plaza test w/ exits" (from_json ho) "ho plaza"
      [
        "southwest";
        "south west";
        "Cornell Health";
        "Gannett";
        "chimes";
        "concert";
        "clock tower";
      ];
    exits_test "exits ho plaza test w/ exits and changed order" (from_json ho)
      "ho plaza"
      [
        "south west";
        "southwest";
        "Cornell Health";
        "Gannett";
        "chimes";
        "concert";
        "clock tower";
      ];
    (*Tests for next_room*)
    next_room_test "ho plaza next room taking exit higher from room tower"
      (from_json ho) "tower" "higher" "nirvana";
    next_room_test "ho plaza next room taking exit Ho Plaza from room health"
      (from_json ho) "health" "Ho Plaza" "ho plaza";
    ( "next_room exception Unknownroom" >:: fun _ ->
      assert_raises (UnknownRoom "This room doesn't exist") (fun () ->
          next_room (from_json ho) "hoplaza" "southwest") );
    ( "next_room exception Unknownexit" >:: fun _ ->
      assert_raises (UnknownExit "This exit doesn't exist") (fun () ->
          next_room (from_json ho) "ho plaza" "SouthWest") );
    ( "next_room exception UnknownExit for lonely adventure" >:: fun _ ->
      assert_raises (UnknownExit "This exit doesn't exist") (fun () ->
          next_room (from_json lonely) "the room" "SouthWest") );
    (*Tests for next_rooms*)
    next_rooms_test "next_rooms for room with no exit" (from_json lonely)
      "the room" [];
    next_rooms_test "next_rooms ho plaza from room ho plaza" (from_json ho)
      "ho plaza" [ "health"; "tower" ];
    next_rooms_test "next_rooms ho plaza from health room" (from_json ho)
      "health" [ "ho plaza" ];
    ( "next_rooms UnknownRoom ho plaza" >:: fun _ ->
      assert_raises (UnknownRoom "This room doesn't exist") (fun () ->
          next_rooms (from_json ho) "towers") );
    ( "next_rooms UnknownRoom lonley" >:: fun _ ->
      assert_raises (UnknownRoom "This room doesn't exist") (fun () ->
          next_rooms (from_json lonely) "") );
  ]

(** [parse_test name input expected_output] constructs an OUnit test named
    [name] that asserts the quality of [expected_output] with [parse input]. *)
let parse_test (name : string) (input : string) (expected_output : command) :
    test =
  name >:: fun _ -> assert_equal expected_output (parse input)

let command_tests =
  [
    parse_test "parse w/o extra space" "go ho plaza" (Go [ "ho"; "plaza" ]);
    parse_test "parse go with extra space" "  go   ho   plaza"
      (Go [ "ho"; "plaza" ]);
    parse_test "parse wuit with extra space" "    quit   " Quit;
    parse_test "parse go with extra space2" " go  john  will   kim  "
      (Go [ "john"; "will"; "kim" ]);
    ( "parse exception Empty string of blanks" >:: fun _ ->
      assert_raises Empty (fun () -> parse "       ") );
    ( "parse exception Empty null string" >:: fun _ ->
      assert_raises Empty (fun () -> parse "") );
    ( "parse exception malformed bad verb" >:: fun _ ->
      assert_raises Malformed (fun () -> parse "jimmy ho plaza") );
    ( "parse exception malformed word after quit" >:: fun _ ->
      assert_raises Malformed (fun () -> parse "quit ppp") );
    ( "parse exception_malformed only blank space after go" >:: fun _ ->
      assert_raises Malformed (fun () -> parse "go  ") );
    ( "parse exception_malformed nothing after go" >:: fun _ ->
      assert_raises Malformed (fun () -> parse "go") );
    ( "parse exception_malformed command after quit with extra whitespace"
    >:: fun _ -> assert_raises Malformed (fun () -> parse "   quit  p") );
  ]

(*These are helpers that create states for the test cases for the States Unit*)
let ho_add1 =
  let i =
    ho |> Adventure.from_json |> init_state
    |> State.go "south west" (from_json ho)
  in
  match i with
  | Legal x -> x
  | Illegal -> init_state (from_json ho)

let ho_illegal =
  let i =
    ho |> Adventure.from_json |> init_state
    |> State.go "should fail room" (from_json ho)
  in
  match i with
  | Legal x -> Legal x
  | Illegal -> Illegal

let ho_add2 =
  let i =
    ho |> Adventure.from_json |> init_state
    |> State.go "south west" (from_json ho)
  in
  match i with
  | Illegal -> init_state (from_json ho)
  | Legal x -> (
      let j = x |> State.go "north east" (from_json ho) in
      match j with
      | Legal y -> y
      | Illegal -> init_state (from_json ho))
(*This last branch should never be hit*)

let state_tests =
  [
    ( "test_current_room_id ho plaza initialized" >:: fun _ ->
      assert_equal
        (current_room_id (State.init_state (Adventure.from_json ho)))
        "ho plaza" );
    ( "test_visitted plaza initialized" >:: fun _ ->
      assert_equal
        (ho |> Adventure.from_json |> init_state |> State.visited)
        [ "ho plaza" ] );
    ( "test_visitted plaza after one go coommand" >:: fun _ ->
      assert_equal ~cmp:cmp_set_like_lists (ho_add1 |> visited)
        [ "ho plaza"; "health" ] ~printer:(pp_list pp_string) );
    ( "test_current_room plaza after one go command" >:: fun _ ->
      assert_equal (ho_add1 |> current_room_id) "health" ~printer:pp_string );
    ("test_ho_illegal_exit" >:: fun _ -> assert_equal ho_illegal Illegal);
    ( "test visitted after 2 go commands are parsed and a room is visitted twice"
    >:: fun _ ->
      assert_equal ~cmp:cmp_set_like_lists (ho_add2 |> visited)
        [ "ho plaza"; "health" ] ~printer:(pp_list pp_string) );
    ( "test current_room after 2 go commands are parsed and a room is visitted \
       twice"
    >:: fun _ ->
      assert_equal "ho plaza" (ho_add2 |> current_room_id) ~printer:pp_string );
  ]

let suite =
  "test suite for A2"
  >::: List.flatten [ cmp_demo; adventure_tests; command_tests; state_tests ]

let _ = run_test_tt_main suite
