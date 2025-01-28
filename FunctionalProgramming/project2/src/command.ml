(* Note: You may introduce new code anywhere in this file. *)

type object_phrase = string list

type command =
  | Go of object_phrase
  | Quit

exception Empty
exception Malformed

let check_empty str =
  List.filter (fun x -> x <> "") (String.split_on_char ' ' str) = []

let check_malformed str_list =
  match str_list with
  | [] -> raise Empty
  | h :: t ->
      if h <> "quit" && h <> "go" then true
      else if h = "quit" && t <> [] then true
      else if h = "go" && t = [] then true
      else false

let parse str =
  if check_empty str then raise Empty
  else if
    check_malformed
      (List.filter (fun x -> x <> "") (String.split_on_char ' ' str))
  then raise Malformed
  else
    match List.filter (fun x -> x <> "") (String.split_on_char ' ' str) with
    | [] -> raise Empty
    | h :: t -> if h = "go" then Go t else Quit
