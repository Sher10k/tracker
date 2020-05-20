/** THIS IS AN AUTOMATICALLY GENERATED FILE.  DO NOT MODIFY
 * BY HAND!!
 *
 * Generated by zcm-gen
 **/

#include <zcm/zcm_coretypes.h>

#ifndef __ZcmLidarScalaObject_hpp__
#define __ZcmLidarScalaObject_hpp__

#include "ZcmLidarScalaUnfilteredObjectAttributes.hpp"
#include "ZcmLidarScalaFilteredObjectAttributes.hpp"


/**
 * @struct ZcmLidarScalaObject
 * @brief Сообщение содержит информацию об объекте
 * @var ZcmLidarScalaObject::objectId
 * @brief - уникальный идентификатор объекта
 * @var ZcmLidarScalaObject::unfilteredAttrib
 * @brief - сырые не отфильтрованные параметры
 * @var ZcmLidarScalaObject::filteredAttrib
 * @brief - параметры полученные от отслеживающей модели. \n
 * 			Данные параметры будут получены даже если объект в данный не виден, \n
 * 			но его существование все еще предсказывается.
 *
 */
class ZcmLidarScalaObject
{
    public:
        int64_t    objectId;

        ZcmLidarScalaUnfilteredObjectAttributes unfilteredAttrib;

        ZcmLidarScalaFilteredObjectAttributes filteredAttrib;

    public:
        /**
         * Destructs a message properly if anything inherits from it
        */
        virtual ~ZcmLidarScalaObject() {}

        /**
         * Encode a message into binary form.
         *
         * @param buf The output buffer.
         * @param offset Encoding starts at thie byte offset into @p buf.
         * @param maxlen Maximum number of bytes to write.  This should generally be
         *  equal to getEncodedSize().
         * @return The number of bytes encoded, or <0 on error.
         */
        inline int encode(void* buf, uint32_t offset, uint32_t maxlen) const;

        /**
         * Check how many bytes are required to encode this message.
         */
        inline uint32_t getEncodedSize() const;

        /**
         * Decode a message from binary form into this instance.
         *
         * @param buf The buffer containing the encoded message.
         * @param offset The byte offset into @p buf where the encoded message starts.
         * @param maxlen The maximum number of bytes to reqad while decoding.
         * @return The number of bytes decoded, or <0 if an error occured.
         */
        inline int decode(const void* buf, uint32_t offset, uint32_t maxlen);

        /**
         * Retrieve the 64-bit fingerprint identifying the structure of the message.
         * Note that the fingerprint is the same for all instances of the same
         * message type, and is a fingerprint on the message type definition, not on
         * the message contents.
         */
        inline static int64_t getHash();

        /**
         * Returns "ZcmLidarScalaObject"
         */
        inline static const char* getTypeName();

        // ZCM support functions. Users should not call these
        inline int      _encodeNoHash(void* buf, uint32_t offset, uint32_t maxlen) const;
        inline uint32_t _getEncodedSizeNoHash() const;
        inline int      _decodeNoHash(const void* buf, uint32_t offset, uint32_t maxlen);
        inline static uint64_t _computeHash(const __zcm_hash_ptr* p);
};

int ZcmLidarScalaObject::encode(void* buf, uint32_t offset, uint32_t maxlen) const
{
    uint32_t pos = 0;
    int thislen;
    int64_t hash = (int64_t)getHash();

    thislen = __int64_t_encode_array(buf, offset + pos, maxlen - pos, &hash, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = this->_encodeNoHash(buf, offset + pos, maxlen - pos);
    if (thislen < 0) return thislen; else pos += thislen;

    return pos;
}

int ZcmLidarScalaObject::decode(const void* buf, uint32_t offset, uint32_t maxlen)
{
    uint32_t pos = 0;
    int thislen;

    int64_t msg_hash;
    thislen = __int64_t_decode_array(buf, offset + pos, maxlen - pos, &msg_hash, 1);
    if (thislen < 0) return thislen; else pos += thislen;
    if (msg_hash != getHash()) return -1;

    thislen = this->_decodeNoHash(buf, offset + pos, maxlen - pos);
    if (thislen < 0) return thislen; else pos += thislen;

    return pos;
}

uint32_t ZcmLidarScalaObject::getEncodedSize() const
{
    return 8 + _getEncodedSizeNoHash();
}

int64_t ZcmLidarScalaObject::getHash()
{
    static int64_t hash = _computeHash(NULL);
    return hash;
}

const char* ZcmLidarScalaObject::getTypeName()
{
    return "ZcmLidarScalaObject";
}

int ZcmLidarScalaObject::_encodeNoHash(void* buf, uint32_t offset, uint32_t maxlen) const
{
    uint32_t pos = 0;
    int thislen;

    thislen = __int64_t_encode_array(buf, offset + pos, maxlen - pos, &this->objectId, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = this->unfilteredAttrib._encodeNoHash(buf, offset + pos, maxlen - pos);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = this->filteredAttrib._encodeNoHash(buf, offset + pos, maxlen - pos);
    if(thislen < 0) return thislen; else pos += thislen;

    return pos;
}

int ZcmLidarScalaObject::_decodeNoHash(const void* buf, uint32_t offset, uint32_t maxlen)
{
    uint32_t pos = 0;
    int thislen;

    thislen = __int64_t_decode_array(buf, offset + pos, maxlen - pos, &this->objectId, 1);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = this->unfilteredAttrib._decodeNoHash(buf, offset + pos, maxlen - pos);
    if(thislen < 0) return thislen; else pos += thislen;

    thislen = this->filteredAttrib._decodeNoHash(buf, offset + pos, maxlen - pos);
    if(thislen < 0) return thislen; else pos += thislen;

    return pos;
}

uint32_t ZcmLidarScalaObject::_getEncodedSizeNoHash() const
{
    uint32_t enc_size = 0;
    enc_size += __int64_t_encoded_array_size(NULL, 1);
    enc_size += this->unfilteredAttrib._getEncodedSizeNoHash();
    enc_size += this->filteredAttrib._getEncodedSizeNoHash();
    return enc_size;
}

uint64_t ZcmLidarScalaObject::_computeHash(const __zcm_hash_ptr* p)
{
    const __zcm_hash_ptr* fp;
    for(fp = p; fp != NULL; fp = fp->parent)
        if(fp->v == ZcmLidarScalaObject::getHash)
            return 0;
    const __zcm_hash_ptr cp = { p, (void*)ZcmLidarScalaObject::getHash };

    uint64_t hash = (uint64_t)0x7efedd31653f42d9LL +
         ZcmLidarScalaUnfilteredObjectAttributes::_computeHash(&cp) +
         ZcmLidarScalaFilteredObjectAttributes::_computeHash(&cp);

    return (hash<<1) + ((hash>>63)&1);
}

#endif
