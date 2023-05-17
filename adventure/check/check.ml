open Game

module type AdventureSig = sig
  type t

  exception UnknownRoom of string
  exception UnknownExit of string

  val from_json : Yojson.Basic.t -> t
  val start_room : t -> string
  val room_ids : t -> string list
  val description : t -> string -> string
  val exits : t -> string -> string list
  val next_room : t -> string -> string -> string
  val next_rooms : t -> string -> string list
end

module AdventureCheck : AdventureSig = Adventure

module type CommandSig = sig
  type object_phrase = string list

  type command =
    | Go of object_phrase
    | Quit

  exception Empty
  exception Malformed

  val parse : string -> command
end

module CommandCheck : CommandSig = Command

module type StateSig = sig
  type t

  val init_state : Adventure.t -> t
  val current_room_id : t -> string
  val visited : t -> string list

  type result =
    | Legal of t
    | Illegal

  val go : string -> Adventure.t -> t -> result
end

module StateCheck : StateSig = State

module type AuthorSig = sig
  val hours_worked : int
end

module AuthorCheck : AuthorSig = Author

let _ = if Author.hours_worked < 0 then exit 1
