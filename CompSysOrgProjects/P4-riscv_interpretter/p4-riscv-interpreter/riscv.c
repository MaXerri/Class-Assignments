#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "linkedlist.h"
#include "hashtable.h"
#include "riscv.h"

/************** BEGIN HELPER FUNCTIONS PROVIDED FOR CONVENIENCE ***************/
const int R_TYPE = 0;
const int I_TYPE = 1;
const int MEM_TYPE = 2;
const int U_TYPE = 3;
const int B_TYPE = 4;
const int UNKNOWN_TYPE = 5;

/**
 * Return the type of instruction for the given operation
 * Available options are R_TYPE, I_TYPE, MEM_TYPE, U_TYPE, B_TYPE, UNKNOWN_TYPE
 */
static int get_op_type(char *op)
{
    const char *r_type_op[] = {"add", "sub", "and", "slt", "sll", "sra"};
    const char *i_type_op[] = {"addi", "andi"};
    const char *mem_type_op[] = {"lw", "lb", "sw", "sb"};
    const char *u_type_op[] = {"lui"};
    const char *b_type_op[] = {"beq"};
    for (int i = 0; i < (int)(sizeof(r_type_op) / sizeof(char *)); i++)
    {
        if (strcmp(r_type_op[i], op) == 0)
        {
            return R_TYPE;
        }
    }
    for (int i = 0; i < (int)(sizeof(i_type_op) / sizeof(char *)); i++)
    {
        if (strcmp(i_type_op[i], op) == 0)
        {
            return I_TYPE;
        }
    }
    for (int i = 0; i < (int)(sizeof(mem_type_op) / sizeof(char *)); i++)
    {
        if (strcmp(mem_type_op[i], op) == 0)
        {
            return MEM_TYPE;
        }
    }
    for (int i = 0; i < (int)(sizeof(u_type_op) / sizeof(char *)); i++)
    {
        if (strcmp(u_type_op[i], op) == 0)
        {
            return U_TYPE;
        }
    }
    for (int i = 0; i < (int)(sizeof(b_type_op) / sizeof(char *)); i++)
    {
        if (strcmp(b_type_op[i], op) == 0)
        {
            return B_TYPE;
        }
    }
    return UNKNOWN_TYPE;
}
/*************** END HELPER FUNCTIONS PROVIDED FOR CONVENIENCE ****************/

registers_t *registers;
char **program;
int no_of_instructions;
// TODO: create any additional variables to store the state of the interpreter
int pc;
hashtable_t *memory;

void init(registers_t *starting_registers, char **input_program, int given_no_of_instructions)
{
    registers = starting_registers;
    program = input_program;
    no_of_instructions = given_no_of_instructions;
    // TODO: initialize any additional variables needed for state
    int pc = 0;
    memory = ht_init(500); // i think this is right?
    registers->r[0] = 0;   // set x0 to 0
}

void end()
{
    // TODO: Free everything from memory
    ht_free(memory); // free the memory allocated to memory hashtable
    for (int i = 0; i < no_of_instructions; i++)
    {
        free(program[i]);
    }
    free(program);
    free(registers);
}

// TODO: create any necessary helper functions

/**
 * return the byte (either 0,1,2, or 3) of int i as an integer
 */
int byte(int i, int byte_num)
{
    if (byte_num == 0)
    {
        return 255 & i;
    }
    else if (byte_num == 1)
    {
        return (int)((unsigned int)(65280 & i) >> 8);
    }
    else if (byte_num == 2)
    {
        return (int)((unsigned int)(16711680 & i) >> 16);
    }
    else if (byte_num == 3)
    {
        return (int)((unsigned int)(4278190080 & i) >> 24);
    }
    else
    {
        return 0;
    }
}

/**
 * return an integer sign extended immediate for both integer and hex immediates
 */
int immed_to_int(char *imm, int op)
{
    char *str[sizeof(imm)];
    removeLeading(imm, str);
    int immediate = 0;
    if (strncmp(str, "0x", 2) == 0)
    {
        immediate = (int)strtol(imm, NULL, 16);
    }
    else
    {
        immediate = atoi(imm);
    }

    immediate = s_ext_imm(immediate, op); // sign extend immediate
    return immediate;
}

/**
 * hlper function for the immed_to_int function which performs the sign extension
 */
int s_ext_imm(int imm, int op_num)
{

    int ans = 0;
    if (op_num == 1 || op_num == 2)
    {
        ans = imm & 0b00000000000000000000111111111111;
        // Checking if sign-bit of number is 0 or 1
        if (imm & 0b100000000000)
        {
            // If number is negative, append leading 1's to the sequence
            ans = ans | 0b11111111111111111111000000000000;
        }
    }
    else if (op_num == 3)
    {
        ans = imm & 0b00000000000011111111111111111111;
        // Checking if sign-bit of number is 0 or 1
        if (imm & 0b10000000000000000000)
        {
            // If number is negative, append leading 1's to the sequence
            ans = ans | 0b11111111111100000000000000000000;
        }
    }
    else if (op_num == 4) // beq uses 13 bit immediate
    {
        ans = imm & 0b00000000000000000001111111111111;
        // Checking if sign-bit of number is 0 or 1
        if (imm & 0b1000000000000)
        {
            // If number is negative, append leading 1's to the sequence
            ans = ans | 0b11111111111111111110000000000000;
        }
    }
    else
    {
        ans = 0;
    }
    return ans;
}

/**
 * sign extends the 8 bit num loaded from memory by the lb instruction
 */
int s_ext_lb(int n)
{
    int ans;
    ans = n & 0b00000000000000000000000011111111;
    // Checking if sign-bit of number is 0 or 1
    if (n & 0b10000000)
    {
        // If number is negative, append leading 1's to the sequence
        ans = ans | 0b11111111111111111111111100000000;
    }
    return ans;
}

/**
 * removes leading spaces from string and returns the new pointer to the string
 */
void removeLeading(char *str, char *str1)
{
    int idx = 0, j, k = 0;

    while (str[idx] == ' ' || str[idx] == '\t' || str[idx] == '\n')
    {
        idx++;
    }

    for (j = idx; str[j] != '\0'; j++)
    {
        str1[k] = str[j];
        k++;
    }

    str1[k] = '\0';
}

void step(char *instruction)
{
    // Extracts and returns the substring before the first space character,
    // by replacing the space character with a null-terminator.
    // `instruction` is MODIFIED IN PLACE to point to the next character
    // after the space.
    // See `man strsep` for how this library function works.

    char *op = strsep(&instruction, " "); // You're free to modify this line.
    // Uses the provided helper function to determine the type of instruction
    int op_type = get_op_type(op);

    // TODO: write logic for evaluating instruction on current interpreter state
    int immediate = 0;

    if (op_type == 0)
    {
        char *rd = strsep(&instruction, ",");
        char *r1 = strsep(&instruction, ",");
        char *r2 = instruction;

        int scrap1 = atoi(strsep(&rd, "x"));
        int scrap2 = atoi(strsep(&r1, "x"));
        int scrap3 = atoi(strsep(&r2, "x"));

        int rd_i = atoi(rd);
        int r1_i = atoi(r1);
        int r2_i = atoi(r2);

        char *add = "add";
        char *sub = "sub";
        char *and = "and";
        char *slt = "slt";
        char *sll = "sll";
        char *sra = "sra";

        if (strcmp(op, add) == 0)
        {
            registers->r[rd_i] = registers->r[r1_i] + registers->r[r2_i];
        }
        else if (strcmp(op, sub) == 0)
        {
            registers->r[rd_i] = registers->r[r1_i] - registers->r[r2_i];
        }
        else if (strcmp(op, and) == 0)
        {
            registers->r[rd_i] = registers->r[r1_i] & registers->r[r2_i];
        }
        else if (strcmp(op, slt) == 0)
        {
            if (registers->r[r1_i] < registers->r[r2_i])
            {
                registers->r[rd_i] = 1;
            }
            else
            {
                registers->r[rd_i] = 0;
            }
        }
        else if (strcmp(op, sll) == 0)
        {
            registers->r[rd_i] = registers->r[r1_i] << registers->r[r2_i];
        }
        else if (strcmp(op, sra) == 0)
        {
            registers->r[rd_i] = registers->r[r1_i] >> registers->r[r2_i];
        }
        else
        {
            registers->r[0] = 0;
        }
        pc = pc + 1;
    }
    else if (op_type == 1)
    {
        char *addi = "addi";
        char *andi = "andi";

        char *rd = strsep(&instruction, ",");
        char *r1 = strsep(&instruction, ",");
        char *imm = instruction;

        int sc1 = atoi(strsep(&rd, "x"));
        int sc2 = atoi(strsep(&r1, "x"));
        int rd_i = atoi(rd);
        int r1_i = atoi(r1);

        immediate = immed_to_int(imm, 1); // format immediate

        // do computation for different instructions
        if (strcmp(op, addi) == 0)
        {

            registers->r[rd_i] = registers->r[r1_i] + immediate;
        }
        else if (strcmp(op, andi) == 0)
        {
            registers->r[rd_i] = registers->r[r1_i] & immediate;
        }
        else
        {
            registers->r[0] = 0;
        }

        pc = pc + 1;
    }
    else if (op_type == 2) // load and stores
    {
        char *lw = "lw";
        char *lb = "lb";
        char *sw = "sw";
        char *sb = "sb";

        char *rd = strsep(&instruction, ",");
        char *imm = strsep(&instruction, "(");
        char *r1 = strsep(&instruction, ")");

        int sc1 = atoi(strsep(&rd, "x"));
        int sc2 = atoi(strsep(&r1, "x"));

        int rd_i = atoi(rd);
        int r1_i = atoi(r1);

        immediate = immed_to_int(imm, 2); // format immediate

        // perform instructions for the loads and stores
        if (strcmp(op, sw) == 0)
        {
            for (int i = 0; i < 4; i++)
            {
                ht_add(memory, immediate + registers->r[r1_i] + i,
                       byte(registers->r[rd_i], i));
            }
        }
        else if (strcmp(op, sb) == 0)
        {
            // edit this TODO
            ht_add(memory, immediate + registers->r[r1_i], byte(registers->r[rd_i], 0));
        }
        else if (strcmp(op, lw) == 0)
        {
            int total = 0;
            for (int i = 0; i < 4; i++)
            {
                total += ht_get(memory, immediate + registers->r[r1_i] + i) << (8 * i);
            }
            registers->r[rd_i] = total;
        }
        else if (strcmp(op, lb) == 0)
        {
            int x = s_ext_lb(ht_get(memory, immediate + registers->r[r1_i]));
            registers->r[rd_i] = x;
        }
        else
        {
            registers->r[0] = 0;
        }
        pc = pc + 1;
    }
    else if (op_type == 3) // lui
    {
        char *lui = "lui";

        char *rd = strsep(&instruction, ",");
        char *imm = strsep(&instruction, ",");

        int sc1 = atoi(strsep(&rd, "x"));

        int rd_i = atoi(rd);

        immediate = immed_to_int(imm, 3); // format immediate

        if (strcmp(op, lui) == 0)
        {
            registers->r[rd_i] = immediate << 12;
        }
        pc = pc + 1;
    }

    else if (op_type == 4) // beq
    {
        char *beq = "beq";

        char *r1 = strsep(&instruction, ",");
        char *r2 = strsep(&instruction, ",");
        char *imm = instruction;

        int sc1 = atoi(strsep(&r1, "x"));
        int sc2 = atoi(strsep(&r2, "x"));

        int r1_i = atoi(r1);
        int r2_i = atoi(r2);

        immediate = immed_to_int(imm, 4); // format immediate

        if (strcmp(op, beq) == 0)
        {
            if (registers->r[r1_i] == registers->r[r2_i])
            {
                pc = pc + (immediate / 4); // change pc counter for beq instruction
                // printf("%d\n", immediate / 4);
            }
            else
            {
                pc = pc + 1;
            }
        }
    }
    else
    {
        registers->r[0] = 0;
    }
    registers->r[0] = 0; // ensure that register zero stays zero
}

void evaluate_program()
{
    // TODO: write logic for evaluating the program
    while (pc < no_of_instructions)
    {
        char *instr = program[pc];
        char *copy = malloc(sizeof(instr)); // need to make copy to not alter instr
        strcpy(copy, instr);
        step(copy);
        free(copy); // free the memory used to store the copied instruction
    }
}
