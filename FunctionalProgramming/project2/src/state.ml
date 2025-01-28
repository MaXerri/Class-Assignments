(* Note: You may introduce new code anywhere in this file. *)

type t = {
  current_room : string;
  visitted_rooms : string list;
}

open Yojson.Basic.Util

(* let exit_of_json j = { name = j |> member "name" |> to_string; room_id = j |>
   member "room id" |> to_string; }

   let room_of_json j = { id = j |> member "id" |> to_string; exits = j |>
   member "exits" |> to_list |> List.map exit_of_json; }

   let t_of_json j = { current_room = j |> member "start room" |> to_string;
   visitted_rooms = (j |> member "start room" |> to_string) :: []; } *)

let init_state adv =
  {
    current_room = Adventure.start_room adv;
    visitted_rooms = [ Adventure.start_room adv ];
  }

let current_room_id st = st.current_room
let visited st = st.visitted_rooms |> List.sort_uniq compare

type result =
  | Legal of t
  | Illegal

let legal ex lst = List.filter (fun i -> i = ex) lst

let go ex adv st =
  if legal ex (Adventure.exits adv st.current_room) = [] then Illegal
  else
    Legal
      {
        current_room = Adventure.next_room adv st.current_room ex;
        visitted_rooms =
          Adventure.next_room adv st.current_room ex :: st.visitted_rooms;
      }
