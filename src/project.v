// ═══════════════════════════════════════════════════════════════
// TinyTapeout Wrapper — MAC Unit AI Inference Accelerator
// ═══════════════════════════════════════════════════════════════
// Top module: tt_um_mariana_mac
//
// PIN MAPPING:
//   ui_in[7:0]  — 8-bit data input (weight or activation)
//   uo_out[7:0] — 8-bit accumulator output (byte-selected)
//   uio[0]      — load_weight (input): latch weight from ui_in
//   uio[1]      — compute (input): multiply weight × ui_in and accumulate
//   uio[2]      — clear (input): clear accumulator
//   uio[3]      — byte_sel (input): 0=acc[7:0], 1=acc[15:8]
//   uio[4]      — overflow (output)
//   uio[5]      — done (output)
//   uio[6]      — sign bit of accumulator (output)
//   uio[7]      — unused
//
// PROTOCOL (2-phase):
//   Phase 1: ui_in=weight, uio[0]=1 → next clock latches weight
//   Phase 2: ui_in=activation, uio[1]=1 → next clock does MAC
//   Read:    uio[3]=0/1 selects output byte
//   Clear:   uio[2]=1 → next clock clears accumulator
// ═══════════════════════════════════════════════════════════════

module tt_um_mariana_mac (
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

    // Control
    wire load_weight = uio_in[0];
    wire compute     = uio_in[1];
    wire clear_acc   = uio_in[2];
    wire byte_sel    = uio_in[3];
    
    // Bidirectional: [3:0]=input, [6:4]=output, [7]=input
    assign uio_oe = 8'b01110000;
    
    // Registers
    reg [7:0]  weight_reg;
    reg [31:0] accumulator;
    reg        overflow_reg;
    reg        done_reg;
    reg        compute_prev;  // for edge detect
    
    // Multiply: weight_reg × ui_in (both signed INT8)
    wire signed [15:0] product = $signed(weight_reg) * $signed(ui_in);
    wire signed [31:0] sum     = $signed(accumulator) + $signed(product);
    
    // Edge detect on compute
    wire compute_rising = compute & ~compute_prev;
    
    // Output mux
    assign uo_out = byte_sel ? accumulator[15:8] : accumulator[7:0];
    
    // Status
    assign uio_out = {1'b0, accumulator[31], done_reg, overflow_reg, 4'b0};
    
    always @(posedge clk) begin
        if (!rst_n) begin
            weight_reg   <= 0;
            accumulator  <= 0;
            overflow_reg <= 0;
            done_reg     <= 0;
            compute_prev <= 0;
        end else if (ena) begin
            compute_prev <= compute;
            done_reg     <= 0;
            
            if (clear_acc) begin
                accumulator  <= 0;
                overflow_reg <= 0;
            end else if (load_weight) begin
                weight_reg <= ui_in;
            end else if (compute_rising) begin
                accumulator  <= sum[31:0];
                done_reg     <= 1;
                overflow_reg <= (accumulator[31] == product[15]) &&
                               (sum[31] != accumulator[31]);
            end
        end
    end

endmodule
