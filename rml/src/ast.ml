(** The abstract syntax tree type. *)

(******************************************************************************
   These types (id, handle, uop, bop) are used by the parser and type-checker.
   You do not want to change them.
 ******************************************************************************)

type id = string
type handle = int

type bop =
  | Add
  | Sub
  | Mul
  | Div
  | Mod
  | And
  | Or
  | Lt
  | Le
  | Gt
  | Ge
  | Eq
  | Ne
  | Cat
  | Pipe
  | Cons
  | Assign
  | Bind

and uop =
  | Neg
  | Not
  | Ref
  | Deref

(******************************************************************************
   [pat] is the type of the AST for patterns. You may implement
   this type however you wish. Look at the formal semantics and think about other
   AST types we have worked with for inspiration.
 ******************************************************************************)

type pat =
  | PUnit
  | PWild
  | PBool of bool
  | PInt of int
  | PString of string
  | PVar of string
  | PPair of pat * pat
  | PNil
  | PCons of pat * pat

(******************************************************************************
   [expr] is the type of the AST for expressions. You may implement
   this type however you wish.  Use the example interpreters seen in
   the textbook as inspiration.
 ******************************************************************************)
(* type 'a sequence = | Nil | Cons of 'a * (unit -> 'a sequence) *)

type expr =
  | Unit of unit
  | Int of int
  | String of string
  | Bool of bool
  | Var of string
  | Pair of expr * expr
  | List of expr list
  | Uop of uop * expr
  | Bop of bop * expr * expr
  | If of expr * expr * expr
  | Sequence of expr * expr
  | Let of pat * expr * expr
  | Func of pat * expr
  | Fapp of expr * expr
  | Pat of expr * (pat * expr) list
(*| Ref of expr | Deref of expr ref | Assign of expr * expr *)

(******************************************************************************
   [defn] is the type of the AST for definitions. You may implement this type
   however you wish.  There are only two kinds of definition---the let
   definition and the let [rec] definition---so this type can be quite simple.
 ******************************************************************************)
and defn = LetD of pat * expr

(******************************************************************************
   [prog] is the type of the AST for an RML program. You should 
   not need to change it.
 ******************************************************************************)

type prog = defn list
