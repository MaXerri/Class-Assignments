open OUnit2
open Search
open ListDictionary
(*****************************************************************)
(* Examples of how to create data structures *)
(*****************************************************************)

module Int = struct
  type t = int

  let compare x y =
    match Stdlib.compare x y with
    | x when x < 0 -> Dictionary.LT
    | 0 -> EQ
    | _ -> GT

  let to_string = string_of_int
end

(* Example: A list dictionary that maps ints to ints. *)
module D1 = ListDictionary.Make (Int) (Int)

(* Example: A set of strings, implemented with list dictionaries. *)
module S1 = DictionarySet.Make (StringKey.String) (ListDictionary.Make)

(* Example: A tree dictionary that maps case-insensitive strings to ints. *)
module D2 = TreeDictionary.Make (StringKey.CaselessString) (Int)

(* Example: A tree dictionary that maps ints to ints*)
module D3 = TreeDictionary.Make (Int) (Int)

(* Example: A set of strings, implemented with tree dictionaries. *)
module S2 = DictionarySet.Make (StringKey.String) (TreeDictionary.Make)

(*****************************************************************)
(* Examples of how to index a directory *)
(*****************************************************************)

let data_dir_prefix = "data" ^ Filename.dir_sep
let preamble_path = data_dir_prefix ^ "preamble"

let preamble_list_idx =
  try Some (ListEngine.E.index_of_dir preamble_path) with _ -> None

let preamble_tree_idx =
  try Some (TreeEngine.E.index_of_dir preamble_path) with _ -> None

(*****************************************************************)
(* Test suite *)
(*****************************************************************)

(*string_printer*)
let pp s = s

(*function for helping to test fold*)
let sum_of_pair_minus_acc k v acc = k + v - acc

(*This module type is the type used to pass in if remove should be tested in the
  testing functors*)
module type Bo = sig
  val rem : bool
end

(*Module indicating remove should be tested*)
module T : Bo = struct
  let rem = true
end

(*module indicating remove should not be tested*)
module F : Bo = struct
  let rem = false
end

module DictTester (DM : Dictionary.DictionaryMaker) (B : Bo) = struct
  module D = DM (Int) (Int)

  let insert1 = D.empty |> D.insert 2 2 |> D.insert 1 3 |> D.insert 1 0

  let delete1 =
    if B.rem = true then
      D.empty |> D.insert 2 2 |> D.insert 1 3 |> D.insert 1 0 |> D.insert 5 5
      |> D.remove 1
    else D.empty

  let tests =
    [
      (*Tests for ListDictionary*)
      ("is_empty true case" >:: fun _ -> assert_equal (D.is_empty D.empty) true);
      ( "is_empty false case" >:: fun _ ->
        assert_equal (D.is_empty (D.empty |> D.insert 1 3)) false );
      ("is_empty size test" >:: fun _ -> assert_equal (D1.size D1.empty) 0);
      ( "inserting duplicate keys size check" >:: fun _ ->
        assert_equal (D.size insert1) 2 );
      ( "inserting duplicate keys print check" >:: fun _ ->
        assert_equal (D.to_string insert1) "{1: 0, 2: 2}" ~printer:pp );
      ( "find test for existing key" >:: fun _ ->
        assert_equal (D.find 1 insert1) (Some 0) );
      ( "find test for non existant key" >:: fun _ ->
        assert_equal (D.find 10 insert1) None );
      ( "find test for empty dict" >:: fun _ ->
        assert_equal (D.find 10 D.empty) None );
      ( "find test for key smaller than any existing" >:: fun _ ->
        assert_equal (D.find 0 insert1) None );
      ( "member test for non existant key" >:: fun _ ->
        assert_equal (D.member 10 insert1) false );
      ( "member test for existant key" >:: fun _ ->
        assert_equal (D.member 1 insert1) true );
      ( "fold subtraction test" >:: fun _ ->
        assert_equal
          (D.fold sum_of_pair_minus_acc 0 insert1)
          3 ~printer:string_of_int );
      ( "to_lst test printer " >:: fun _ ->
        assert_equal (D.to_string insert1) "{1: 0, 2: 2}" ~printer:pp );
      ( "to_lst test " >:: fun _ ->
        assert_equal (D.to_list insert1) [ (1, 0); (2, 2) ] );
      (if B.rem = true then
       let delete1 =
         D.empty |> D.insert 2 2 |> D.insert 1 3 |> D.insert 1 0 |> D.insert 5 5
         |> D.remove 1
       in
       "remove non_existant key test" >:: fun _ ->
       assert_equal
         (D.to_string (delete1 |> D.remove 9))
         "{2: 2, 5: 5}" ~printer:pp
      else "dont run" >:: fun _ -> assert_equal 0 0);
    ]
end

module SetTester (DM : Dictionary.DictionaryMaker) (B : Bo) = struct
  module SM = DictionarySet.Make (StringKey.String) (DM)

  let s1 =
    SM.empty |> SM.insert "happy" |> SM.insert "apple" |> SM.insert "zed"
    |> SM.insert "john"

  let s2 =
    SM.empty |> SM.insert "happy" |> SM.insert "apple" |> SM.insert "zed"
    |> SM.insert "mario"

  let s3 =
    SM.empty |> SM.insert "apple" |> SM.insert "happy" |> SM.insert "john"
    |> SM.insert "mario" |> SM.insert "zed"

  let s4 = SM.empty |> SM.insert "apple" |> SM.insert "happy" |> SM.insert "zed"
  let s5 = SM.empty |> SM.insert "john" |> SM.insert "mario"

  let s6 =
    if B.rem then s1 |> SM.insert "happy" |> SM.remove "apple" else SM.empty

  let s7 =
    SM.empty |> SM.insert "zed" |> SM.insert "apple" |> SM.insert "happy"
    |> SM.insert "apple"

  let tests =
    [
      ( "set test empty " >:: fun _ ->
        assert_equal (SM.size SM.empty) 0 ~printer:string_of_int );
      ( "set union test sets not equal" >:: fun _ ->
        assert_equal (SM.to_list (SM.union s1 s2)) (SM.to_list s3) );
      ( "set intersect normal " >:: fun _ ->
        assert_equal (SM.to_list (SM.intersect s1 s2)) (SM.to_list s4) );
      ( "set difference normal " >:: fun _ ->
        assert_equal (SM.to_list (SM.difference s1 s2)) (SM.to_list s5) );
      (*Tests for dict functions*)
      ( "set is_empty true case" >:: fun _ ->
        assert_equal (SM.is_empty SM.empty) true );
      ( "set is_empty false case" >:: fun _ ->
        assert_equal (SM.is_empty (SM.empty |> SM.insert "johnson")) false );
      ("set is_empty size test" >:: fun _ -> assert_equal (S1.size S1.empty) 0);
      (if B.rem then
       "set inserting duplicate keys size check" >:: fun _ ->
       assert_equal (SM.size s6) 3
      else "" >:: fun _ -> assert_equal 0 0);
      ( "set member test for non existant key" >:: fun _ ->
        assert_equal (SM.member "appl" s5) false );
      ( "set member test for existant key" >:: fun _ ->
        assert_equal (SM.member "apple" s1) true );
      (if B.rem then
       "set remove key size test" >:: fun _ -> assert_equal (SM.size s6) 3
      else "" >:: fun _ -> assert_equal 0 0);
      (if B.rem then
       "set dictset print test" >:: fun _ ->
       assert_equal (SM.to_string s6) "[\"happy\"; \"john\"; \"zed\"]"
         ~printer:pp
      else "" >:: fun _ -> assert_equal 0 0);
      (if B.rem then
       "set dictset remove non_existent key from empty list" >:: fun _ ->
       assert_equal
         (SM.to_string (SM.remove "aa" (SM.empty |> SM.insert "aa")))
         "[]" ~printer:pp
      else "" >:: fun _ -> assert_equal 0 0);
      ( "set dictset to_lst test " >:: fun _ ->
        assert_equal (SM.to_list s1) [ "apple"; "happy"; "john"; "zed" ] );
      ( "set dictset to_lst string test " >:: fun _ ->
        assert_equal (SM.to_string s1)
          "[\"apple\"; \"happy\"; \"john\"; \"zed\"]" ~printer:pp );
      ( "set print tree with duplicate" >:: fun _ ->
        assert_equal (SM.to_string s7) "[\"apple\"; \"happy\"; \"zed\"]"
          ~printer:pp );
    ]
end

module ListDictTester = DictTester (ListDictionary.Make) (T)
module TreeDictTester = DictTester (TreeDictionary.Make) (F)
module TreeSetTester = SetTester (TreeDictionary.Make) (F)
module DictSetTester = SetTester (ListDictionary.Make) (T)

(*Scalability test for TreeDict; runtime actual ounit test here is irrelevant
  and i do not run this test in the suite *)
let scalability_fold_and_insert =
  [
    (let dict = ref D3.empty in
     let dict_lst = ref [] in
     for x = 1 to 1000000 do
       dict := D3.insert (Random.int 10000) (Random.int 100) !dict;
       dict_lst := (Random.int 10000, Random.int 100) :: !dict_lst
     done;
     let combine = D3.fold sum_of_pair_minus_acc 0 !dict in
     "scalability fold " >:: fun _ ->
     assert_equal 0 combine ~printer:string_of_int);
  ]

let suite =
  "search test suite"
  >::: List.flatten
         [
           ListDictTester.tests;
           TreeDictTester.tests;
           DictSetTester.tests;
           TreeSetTester.tests (*scalability_fold_and_insert;*);
         ]

let _ = run_test_tt_main suite
