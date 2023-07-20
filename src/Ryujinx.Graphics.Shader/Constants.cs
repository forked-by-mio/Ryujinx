namespace Ryujinx.Graphics.Shader
{
    static class Constants
    {
        public const int ConstantBufferSize = 0x10000; // In bytes

        public const int MaxAttributes = 16;
        public const int AllAttributesMask = (int)(uint.MaxValue >> (32 - MaxAttributes));

        public const int NvnBaseVertexByteOffset = 0x640;
        public const int NvnBaseInstanceByteOffset = 0x644;
        public const int NvnDrawIndexByteOffset = 0x648;

        // Transform Feedback emulation.

        public const int TfeInfoBinding = 0;
        public const int TfeBufferBaseBinding = 1;
        public const int TfeBuffersCount = 4;

        // Bindless emulation.

        public const int BindlessTextureSetIndex = 4;
        public const int BindlessTableBinding = 0;
        public const int BindlessScalesBinding = 1;
        public const int BindlessTableKey = (BindlessTextureSetIndex << 16) | BindlessTableBinding;
        public const int BindlessScalesKey = (BindlessTextureSetIndex << 16) | BindlessScalesBinding;
    }
}
