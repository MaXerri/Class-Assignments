(* Note: You may introduce new code anywhere in this file. *)

exception UnknownRoom of string
exception UnknownExit of string

type exit = {
  name : string;
  room_id : string;
}

type room = {
  id : string;
  description : string;
  exits : exit list;
}

type t = {
  rooms : room list;
  start_room : string;
}

open Yojson.Basic.Util

let exit_of_json j =
  {
    name = j |> member "name" |> to_string;
    room_id = j |> member "room id" |> to_string;
  }

let room_of_json j =
  {
    id = j |> member "id" |> to_string;
    description = j |> member "description" |> to_string;
    exits = j |> member "exits" |> to_list |> List.map exit_of_json;
  }

let t_of_json j =
  {
    rooms = j |> member "rooms" |> to_list |> List.map room_of_json;
    start_room = j |> member "start room" |> to_string;
  }

let from_json json = t_of_json json
let start_room adv = adv.start_room

(* helper function which returns the id of a room. i is a room *)
let room_ids_helper i = i.id
let room_ids adv = List.map room_ids_helper adv.rooms |> List.sort_uniq compare

(* helper function which returns the room associated with a room id or a unknown
   room error if the room doesnt exist *)
let rec room_helper lst room =
  match lst with
  | [] -> raise (UnknownRoom "This room doesn't exist")
  | h :: t -> if h.id = room then h else room_helper t room

let description adv room = (room_helper adv.rooms room).description

let exits adv room =
  List.map (fun i -> i.name) (room_helper adv.rooms room).exits
  |> List.sort_uniq compare

(* helper function which returns the exit associated with ex*)
let rec exit_helper exits_lst ex =
  match exits_lst with
  | [] -> raise (UnknownExit "This exit doesn't exist")
  | h :: t -> if h.name = ex then h else exit_helper t ex

let next_room adv room ex =
  (exit_helper (room_helper adv.rooms room).exits ex).room_id

let next_rooms adv room =
  (room |> room_helper adv.rooms).exits
  |> List.map (fun i -> i.room_id)
  |> List.sort_uniq compare
