(** Representation of static adventure data.

    This module represents the data stored in adventure files, including the
    rooms and exits. It handles loading of that data from JSON as well as
    querying the data.

    For examples, the specifications in this interface reference the example "Ho
    Plaza" adventure found in [data/ho_plaza.json]. *)

(**********************************************************************
 * DO NOT CHANGE THIS FILE
 * It is part of the interface the course staff will use to test your
 * submission.
 **********************************************************************)

type t
(** The abstract type of values representing adventures. *)

exception UnknownRoom of string
(** Raised when an unknown room identifier is encountered. It carries the
    identifier of the unknown room. *)

exception UnknownExit of string
(** Raised when an unknown exit is encountered. It carries the name of the
    unknown exit. *)

val from_json : Yojson.Basic.t -> t
(** [from_json j] is the adventure that [j] represents. Requires: [j] is a valid
    JSON adventure representation. *)

val start_room : t -> string
(** [start_room a] is the identifier of the starting room in adventure [a].
    Example: the [start_room] of Ho Plaza is ["ho plaza"]. *)

val room_ids : t -> string list
(** [room_ids a] is a set-like list of all of the room identifiers in adventure
    [a]. Example: the [room_ids] of Ho Plaza are ["ho plaza"], ["health"],
    ["tower"], and ["nirvana"]. *)

val description : t -> string -> string
(** [description a r] is the description of the room with identifier [r] in
    adventure [a]. Raises [UnknownRoom r] if [r] is not a room identifier in
    [a]. Example: in Ho Plaza, the [description] of room identifier ["ho plaza"]
    begins with ["You are on Ho Plaza. "] and continues with a few more
    sentences, which are omitted here for brevity. *)

val exits : t -> string -> string list
(** [exits a r] is a set-like list of all exit names from the room with
    identifier [r] in adventure [a]. Raises [UnknownRoom r] if [r] is not a room
    identifier in [a]. Example: in Ho Plaza, the [exits] from room identifier
    ["health"] are ["northeast"], ["north east"], and ["Ho Plaza"]. *)

val next_room : t -> string -> string -> string
(** [next_room a r e] is the identifier of the room in adventure [a] that is
    immediately reached by taking the exit named [e] from the room with
    identifier [r]. Raises [UnknownRoom r] if [r] is not a room identifier in
    [a]. Raises [UnknownExit e] if [e] is not the name of an exit from the room
    with identifier [r] in [a]. Example: in Ho Plaza, taking exit ["northeast"]
    from room identifier ["health"] results in the [next_room] of ["ho plaza"]. *)

val next_rooms : t -> string -> string list
(** [next_rooms a r] is a set-like list of all the identifiers of rooms in
    adventure [a] that are immediately reachable by taking any exit from the
    room with identifier [r]. Raises [UnknownRoom r] if [r] is not a room
    identifier in [a]. Example: in Ho Plaza, the [next_rooms] from room
    identifier ["ho plaza"] are ["health"] and ["tower"]. *)
